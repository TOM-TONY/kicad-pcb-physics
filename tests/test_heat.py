import json
import math
import unittest

from kicad_pcb_physics.heat import analyze


def rectangle(x0, y0, x1, y1):
    return {"shell": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": []}


def coupon():
    rect = rectangle(0, 0, 8, 4)
    layers = ["Front", "Back"]
    return {
        "outline": [rect],
        "layers": layers,
        "parts": {},
        "nets": {
            "plane": {"copper": {l: [rect] for l in layers}, "pads": {}, "vias": []}
        },
    }


class HeatTests(unittest.TestCase):
    def test_parabolic_strip_and_second_order_convergence(self):
        board = coupon()
        sheet = 0.3 * 0.001 / 2 + 390 * 35e-6
        generation = 0.5 / (0.008 * 0.004)
        errors = []
        for pitch in (1, 0.5, 0.25):
            settings = dict(
                grid_mm=pitch,
                material_mm=2,
                dielectric_mm=[1],
                ambient_c=20,
                fixed_edges=["left", "right"],
                h_top_w_m2k=0,
                h_bottom_w_m2k=0,
                layer_power_w={l: 0.5 for l in board["layers"]},
            )
            result = analyze(board, settings, [])
            self.assertAlmostEqual(result["cut_w"], 1, places=9)
            errors.append(
                max(
                    abs(
                        t
                        - 20
                        - generation * (x * 0.001) * (0.008 - x * 0.001) / (2 * sheet)
                    )
                    for x, _, t in result["maps"][0]["points"]
                )
            )
        self.assertAlmostEqual(errors[0] / errors[1], 4, places=7)
        self.assertAlmostEqual(errors[1] / errors[2], 4, places=7)

    def test_through_barrel_and_one_sided_heat_rejection(self):
        board = coupon()
        board["nets"]["plane"]["vias"] = [
            dict(xy=[3, 2], drill_mm=0.5, land_mm=0.8, layers=board["layers"])
        ]
        settings = dict(
            grid_mm=4,
            material_mm=8,
            dielectric_mm=[1],
            via_plating_um=25,
            h_top_w_m2k=0,
            h_bottom_w_m2k=50,
            layer_power_w={"Front": 1},
        )
        got = analyze(board, settings, {})
        cross = math.pi * ((0.25 + 0.025) ** 2 - 0.25**2) * 1e-6
        area = 32e-6
        vertical = 0.3 * (area - cross) / 0.001 + 390 * cross / 0.001035
        self.assertAlmostEqual(
            got["peak_board_c"], 25 + 1 / (50 * area) + 1 / vertical, delta=0.02
        )
        self.assertAlmostEqual(got["balance_error_w"], 0, places=10)

    def test_holes_dc_and_overlapping_pad_geometry(self):
        board = coupon()
        outline = board["outline"][0]
        outline["holes"] = [rectangle(1, 1, 2, 2)["shell"]]
        poly = rectangle(3.1, 1.1, 3.9, 1.9)
        pad = {"layer": "Front", "xy": [3.5, 1.5], "polygons": [poly]}
        board["nets"]["plane"]["pads"] = {"IC.1": [pad, pad]}
        board["parts"] = {
            "IC": {"bounds": [3, 1, 4, 2], "xy": [3.5, 1.5], "layer": "Front"}
        }
        settings = dict(
            grid_mm=0.5, material_mm=1, part_power_w={"IC": 0.2}, contact_k_w=10
        )
        dc = {"load": {"heat_cells": [dict(layer="Back", x_mm=5, y_mm=1, watts=0.05)]}}
        got = analyze(board, settings, dc)
        self.assertEqual(got["contact_model"]["IC"], "pad union")
        self.assertAlmostEqual(got["convected_w"], 0.25)
        self.assertEqual(len(got["maps"][0]["points"]), 124)
        json.dumps(got, allow_nan=False)
        board["nets"]["plane"]["pads"]["IC.1"].pop()
        self.assertAlmostEqual(
            analyze(board, settings, dc)["body_c"]["IC"], got["body_c"]["IC"]
        )

    def test_unsunk_island_and_off_board_source(self):
        with self.assertRaisesRegex(ValueError, "no heat sink"):
            analyze(coupon(), dict(h_top_w_m2k=0, h_bottom_w_m2k=0), {})
        with self.assertRaisesRegex(ValueError, "no positive area"):
            analyze(
                coupon(),
                {},
                [{"heat_cells": [dict(layer="Front", x_mm=20, y_mm=50, watts=1)]}],
            )


class ConformingHeatTests(unittest.TestCase):
    def test_two_material_strip_on_non_aligned_interface(self):
        board = coupon()
        interface = 3.33
        for layer in board["layers"]:
            board["nets"]["plane"]["copper"][layer] = [rectangle(0, 0, interface, 4)]
        # Both layers are driven equally, so through-thickness heat flow vanishes.
        # Integrate S(x) T'(x) = c - q*x, with T(0) = T(L) = 0.
        a, length = interface * 0.001, 0.008
        left = 0.3 * 0.001 / 2 + 390 * 35e-6
        right = 0.3 * 0.001 / 2
        density = 0.5 / (length * 0.004)
        c = (
            density
            * (a * a / left + (length**2 - a * a) / right)
            / (2 * (a / left + (length - a) / right))
        )

        def exact(x_mm):
            x = x_mm * 0.001
            u = min(x, a)
            return (c * u - density * u * u / 2) / left + (
                c * (x - u) - density * (x * x - u * u) / 2
            ) / right

        errors = []
        for grid in (1, 0.5, 0.25):
            r = analyze(
                board,
                dict(
                    method="conformal",
                    grid_mm=grid,
                    near_parts_mm=grid,
                    dielectric_mm=[1],
                    ambient_c=0,
                    fixed_edges=["left", "right"],
                    h_top_w_m2k=0,
                    h_bottom_w_m2k=0,
                    layer_power_w={"Front": 0.5, "Back": 0.5},
                ),
                [],
            )
            vals = r["maps"][0]["points"]
            self.assertTrue(any(abs(x - interface) < 1e-8 for x, _, _ in vals))
            self.assertLess(
                max(map(abs, r["copper_integration_error_mm2"].values())), 1e-7
            )
            self.assertAlmostEqual(r["cut_w"], 1, places=7)
            errors.append(max(abs(t - exact(x)) for x, _, t in vals))
        self.assertLess(errors[1], errors[0] * 0.45)
        self.assertLess(errors[2], errors[1] * 0.45)

    def test_pad_source_dc_square_hole_and_partial_span_barrel(self):
        board = coupon()
        board["layers"].insert(1, "Middle")
        board["outline"][0]["holes"] = [rectangle(1, 1, 2, 2)["shell"]]
        board["nets"]["plane"]["vias"] = [
            dict(xy=[4, 2], drill_mm=0.3, land_mm=0.6, layers=["Front", "Middle"])
        ]
        board["parts"]["P"] = dict(
            layer="Front", xy=[3, 2], bounds=[2.5, 1.5, 3.5, 2.5]
        )
        p = dict(layer="Front", xy=[3, 2], polygons=[rectangle(2.7, 1.8, 3.3, 2.2)])
        board["nets"]["plane"]["pads"]["P.1"] = [p, p]
        settings = dict(
            method="conformal",
            grid_mm=1,
            near_parts_mm=0.25,
            part_power_w={"P": 0.1},
            part_to_air_k_w={"P": 100},
            dielectric_mm=[0.2, 1],
        )
        dc = {
            "case": {"heat_cells": [dict(layer="Middle", x_mm=2, y_mm=2, watts=0.03)]}
        }
        r = analyze(board, settings, dc)
        self.assertEqual(r["method"], "conformal")
        self.assertEqual(r["contact_model"]["P"], "pad union")
        self.assertAlmostEqual(r["area_mm2"], 31)
        self.assertAlmostEqual(r["convected_w"], 0.13, places=7)
        self.assertEqual(list(r["barrel_integration_error_mm2"]), ["Front to Middle"])
        self.assertTrue(all(m["unit"] == "degC" and m["points"] for m in r["maps"]))
        json.dumps(r, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "contact"):
            analyze(board, {**settings, "part_pad_ids": {"P": ["P.1", "missing"]}}, dc)
