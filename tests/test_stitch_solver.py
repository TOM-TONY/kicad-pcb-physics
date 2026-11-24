"""Analytical coupons and independent direct network insertions for via ranking."""

import json
import math
import unittest

import numpy as np
from kicad_pcb_physics import stitch_solver as solver
from kicad_pcb_physics.mesh import MU, Sheet, slab
from kicad_pcb_physics.paths import rectangle


def rect(x0, y0, x1, y1):
    return {"shell": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": []}


def coupon(full=False, bridge=True):
    layers = ["Top", "Back"]
    data = {
        "layers": layers,
        "nets": {
            "Metal": {
                "copper": {l: [rect(0, 0, 12, 1)] for l in layers},
                "vias": [],
                "pads": {},
                "tracks": [],
            }
        },
    }
    m = {
        "depth_mm": {"Top": 0, "Back": 1.2},
        "copper_mm": {"Top": 0.035, "Back": 0.035},
        "plating_mm": 0.025,
        "rho_ohm_m": 1.72e-8,
    }
    net = data["nets"]["Metal"]
    for pin, layer in zip(("P.1", "Q.1"), layers):
        net["pads"][pin] = [
            {
                "xy": [0.5, 0.5],
                "layer": layer,
                "polygons": [rect(0, 0, 12 if full else 1, 1)],
            }
        ]
    if bridge:
        net["vias"] = [via([11.5, 0.5])]
    s = {
        "net": "Metal",
        "source": "P.1",
        "sink": "Q.1",
        "mesh_mm": 1,
        "drill_mm": 0.3,
        "land_mm": 0.7,
        "max_vias": 3,
        "frequency_hz": 2e6,
        "current_A": 2.5,
        "return_spacing_mm": 0.18,
        "objective": "resistance",
    }
    return data, m, s


def via(xy):
    return {"xy": list(xy), "layers": ["Top", "Back"], "drill_mm": 0.3, "land_mm": 0.7}


def direct(g, m, s, coords, f):
    base = g["nets"]["Metal"]
    amended = {
        **g,
        "nets": {"Metal": {**base, "vias": base["vias"] + [via(x) for x in coords]}},
    }
    sheet = Sheet(amended, m, s["net"], s["mesh_mm"])
    v = sheet.solve(
        [
            {"contacts": [s["source"]], "current_A": 1},
            {"contacts": [s["sink"]], "current_A": -1},
        ],
        f,
        reference_height_mm=s["return_spacing_mm"],
    )[1]
    return v[0] - v[1]


class StitchSolver(unittest.TestCase):
    def test_barrel_bank_closed_form_and_open_circuit(self):
        g, m, s = coupon(full=True, bridge=False)
        coords = [[1.5, 0.5], [4.5, 0.5], [8.5, 0.5]]
        for obj in ("resistance", "potential"):
            result = solver.analyze(g, m, {**s, "objective": obj}, coords)
            self.assertIsNone(result["baseline"])
            self.assertEqual(len(result["rows"]), 3)
            f = 0 if obj == "resistance" else s["frequency_hz"]
            omega = math.tau * f
            perimeter = math.pi * s["drill_mm"]
            z = slab(f, 0.025e-3, m["rho_ohm_m"]) * 1.2 / perimeter
            z += (
                2
                * (slab(f, 0.035e-3, m["rho_ohm_m"]) + 1j * omega * MU * 0.18e-3)
                * 0.12
                / perimeter
            )
            reference = {
                "resistance": z.real,
                "potential": abs(z) * 2.5,
                "inductance": z.imag / omega if omega else 0,
            }[obj]
            for i, r in enumerate(result["rows"], 1):
                self.assertAlmostEqual(r["after"] * i / reference, 1, places=10)
                if f:
                    self.assertAlmostEqual(
                        r["metrics_after"]["port_inductance_H"] * i / (z.imag / omega),
                        1,
                        places=10,
                    )
                np.testing.assert_allclose(
                    r["metrics_after"]["dc_added_span_current_A"],
                    [2.5 / i] * i,
                    rtol=1e-10,
                )
            json.dumps(result, allow_nan=False)

    def test_all_candidates_all_stages_against_independent_rebuilds(self):
        g, m, s = coupon()
        coords = [[1.5, 0.5], [4.5, 0.5], [8.5, 0.5]]
        for obj in ("resistance", "potential"):
            result = solver.analyze(g, m, {**s, "objective": obj}, reversed(coords))
            seen = []
            for index, stage in enumerate(result["stages"]):
                for trial in stage["candidates"]:
                    f = 0 if obj == "resistance" else s["frequency_hz"]
                    z = direct(g, m, s, seen + [trial["xy"]], f)
                    expected = (
                        z.real
                        if obj == "resistance"
                        else (
                            abs(z) * 2.5
                            if obj == "potential"
                            else z.imag / (math.tau * f)
                        )
                    )
                    self.assertAlmostEqual(trial["after"] / expected, 1, places=9)
                if index == len(result["rows"]):
                    self.assertTrue(
                        all(q["improvement"] <= 0 for q in stage["candidates"])
                    )
                    continue
                row = result["rows"][index]
                self.assertEqual(
                    row["after"], min(p["after"] for p in stage["candidates"])
                )
                seen.append(row["xy"])
            self.assertEqual(seen[0], coords[0])
            self.assertEqual(len(seen), 3 if obj == "resistance" else 2)
            self.assertTrue(all(p["points"] for p in result["maps"]))

    def test_sheet_strip_mesh_refinement(self):
        g, m, s = coupon(bridge=False)
        g["nets"]["Metal"]["pads"] = {}
        for x, tag in ((0, "left"), (11, "right")):
            g["nets"]["Metal"]["pads"][tag] = [
                {"layer": "Top", "polygons": [rect(x, 0, x + 1, 1)]}
            ]
        for pitch in (1, 0.5, 0.25):
            spec = {
                **s,
                "source": "left",
                "sink": "right",
                "mesh_mm": pitch,
                "max_vias": 0,
            }
            r = solver.analyze(g, m, spec, [])
            # Distance between innermost rows of cell centres of the two electrodes.
            self.assertAlmostEqual(
                r["baseline"] / (10 + pitch), m["rho_ohm_m"] / 0.035e-3, places=12
            )

    def test_middle_layer_floating_and_unresolved_candidate(self):
        g, m, s = coupon(bridge=False)
        g["layers"].insert(1, "Inner")
        m["copper_mm"]["Inner"] = 0.018
        m["depth_mm"]["Inner"] = 0.5
        g["nets"]["Metal"]["copper"]["Inner"] = [rect(0, 0, 12, 1)]
        result = solver.analyze(g, m, s, [[1.5, 0.5], [8.5, 0.5], [2, 3]])
        self.assertIsNone(result["baseline"])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(len(result["rows"]), 2)
        self.assertLess(result["final"], result["rows"][0]["after"])
        self.assertEqual(len(result["final_metrics"]["dc_added_span_current_A"]), 4)
        for freq in (0, s["frequency_hz"]):
            base = g["nets"]["Metal"]
            used = [{**via(r["xy"]), "layers": g["layers"]} for r in result["rows"]]
            mesh = Sheet(
                {**g, "nets": {"Metal": {**base, "vias": used}}}, m, "Metal", 1
            )
            v = mesh.solve(
                [
                    {"contacts": [s["source"]], "current_A": 1},
                    {"contacts": [s["sink"]], "current_A": -1},
                ],
                freq,
                reference_height_mm=s["return_spacing_mm"],
            )[1]
            key = "ac_potential_V" if freq else "dc_potential_V"
            self.assertAlmostEqual(
                result["final_metrics"][key] / (abs(v[0] - v[1]) * 2.5), 1, places=9
            )

    def test_signal_reference_spread_and_directed_excitation(self):
        g, m, s = coupon()
        g["nets"]["Signal"] = {
            "tracks": [{"a": [0, 2.5], "b": [12, 2.5], "layer": "Top"}]
        }
        coords = [[1.5, 0.5], [4.5, 0.5]]
        passive = solver.analyze(g, m, {**s, "signal_net": "Signal"}, coords)
        for name in ("dc_rms_spread_mm", "ac_rms_spread_mm"):
            self.assertAlmostEqual(passive["final_metrics"][name], 2, places=11)
        path = [[0.5, 0.5, 1.2], [7, 2.5, 1.2], [7, 2.5, 0], [0.5, 0.5, 0]]
        s.update(exciting_path_mm=path, objective="inductance")
        driven = solver.analyze(g, m, s, coords)
        self.assertIsNotNone(driven["baseline_metrics"]["loop_inductance_H"])
        self.assertNotEqual(
            driven["baseline_metrics"]["port_inductance_H"], driven["baseline"]
        )
        chosen = []
        for stage in driven["stages"]:
            for trial in stage["candidates"]:
                # Independent full refactor with each via included in the Sheet geometry.
                ref, m2, _ = coupon()
                ref["nets"]["Metal"]["vias"].extend(
                    via(pt) for pt in chosen + [trial["xy"]]
                )
                check = solver.analyze(ref, m2, {**s, "max_vias": 0}, [])
                self.assertAlmostEqual(trial["after"] / check["baseline"], 1, places=8)
            if len(chosen) < len(driven["rows"]):
                chosen.append(driven["rows"][len(chosen)]["xy"])
        self.assertAlmostEqual(
            driven["final_metrics"]["loop_energy_H"] / driven["final"], 1, places=10
        )
        json.dumps(driven, allow_nan=False)

    def test_closed_rectangular_filament_and_regularized_self(self):
        g, m, s = coupon(bridge=False)
        target = g["nets"]["Metal"]
        target["pads"]["Q.1"] = [
            {"xy": [11.5, 0.5], "layer": "Top", "polygons": [rect(11, 0, 12, 1)]}
        ]
        length, gap, r = 11, 0.8, 0.08
        forward = [[11.5, 0.5, 0], [11.5, 0.5, gap], [0.5, 0.5, gap], [0.5, 0.5, 0]]
        f = s["frequency_hz"]
        spec = {
            **s,
            "objective": "inductance",
            "return_spacing_mm": 0,
            "wire_radius_mm": r,
            "max_vias": 0,
        }
        self.assertIsNone(solver.analyze(g, m, spec, [])["baseline"])
        spec.update(exciting_path_mm=forward, filament_step_mm=0.12)
        z = slab(f, 0.035e-3, m["rho_ohm_m"])
        expected = rectangle(length, gap, r) * 1e-9 + length * z.imag / (math.tau * f)
        out = solver.analyze(g, m, spec, [])
        self.assertLess(abs(out["baseline"] / expected - 1), 0.012)
        meta = out["baseline_metrics"]
        self.assertGreater(meta["passivity"]["minimum_L_H"], 0)
        self.assertAlmostEqual(
            meta["loop_energy_H"] / expected, out["baseline"] / expected, places=10
        )

    def test_multicell_rim_equivalence_with_direct_via(self):
        g, m, s = coupon()
        s.update(mesh_mm=0.25, max_vias=1)
        pt = [2.0, 0.5]
        out = solver.analyze(g, m, s, [pt])
        self.assertAlmostEqual(
            out["final"] / direct(g, m, s, [pt], 0).real, 1, places=8
        )

    def test_reject_nonpassive_candidate_with_positive_series_inductance(self):
        for matrix in (
            np.eye(2) + 1j * np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.array([[1.0, 2.0], [2.0, 1.0]]) + 1j * np.eye(2),
        ):
            self.assertGreater(matrix.sum().imag, 0)
            value, status, _ = solver._passive_loop(matrix, 1.0)
            self.assertIsNone(value)
            self.assertIn("non-passive", status)
        self.assertIsNone(solver._passive_loop(np.eye(2) - 1j * np.eye(2), 1.0)[0])

    def test_clipped_boundary_touch_and_pth_barrel(self):
        g, m, s = coupon()
        s["bounds_mm"] = [0, 0, 12, 2]
        g["nets"]["Metal"]["vias"][0]["kind"] = "plated_pad"
        reference = solver.analyze(g, m, s, [[1.5, 0.5]])
        for polys in g["nets"]["Metal"]["copper"].values():
            polys.append(rect(2, 2, 4, 3))
        result = solver.analyze(g, m, s, [[1.5, 0.5]])
        self.assertAlmostEqual(result["final"], reference["final"], places=12)
        self.assertTrue(result["baseline_metrics"]["dc_connected"])

    def test_island_callback_spacing_and_no_joint_optimality_claim(self):
        covered = {(0.0, 0.0): {1, 2}, (1.0, 0.0): {1}, (3.0, 0.0): {2, 3}}

        def score(points):
            return len({1, 2, 3} - set().union(*(covered[tuple(p)] for p in points)))

        s = {
            "objective": "islands",
            "max_vias": 3,
            "min_spacing_mm": 1.5,
            "island_score": score,
        }
        result = solver.analyze({}, {}, s, covered)
        self.assertEqual([r["xy"] for r in result["rows"]], [[0.0, 0.0], [3.0, 0.0]])
        self.assertEqual(result["rows"][1]["improvement"], 1)
        self.assertEqual(
            [r["xy"] for r in result["stages"][1]["candidates"]], [[3.0, 0.0]]
        )
        self.assertEqual(result["final"], 0)
