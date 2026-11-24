import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kicad_pcb_physics.cross_section import C0, section, modal
from kicad_pcb_physics.line_network import response
from kicad_pcb_physics.line_route import detect, point, route
from kicad_pcb_physics import transmission


class LineTests(unittest.TestCase):
    def test_uniform_sections(self):
        # Hammerstad-Jensen, symmetric stripline elliptic-integral result,
        # and Steer's coupled-microstrip example (per-conductor odd/even).
        coupons = [
            ((0.8224,), 0, 0.4, 0, 0, 4, 49.96238, None),
            ((1.4423896,), 0, 0.5, 0.5, 0, 1, 50.0, None),
            ((0.76, 0.76), 0.2, 0.4, 0, 0, 4, 40.79503, 61.41839),
            ((0.4, 0.4), 0.2, 0.4, 0.4, 0, 4, 41.76149, 57.38409),
            ((0.8224,), 0, 0.4, 0, 0.035, 4, 48.65659, None),
        ]
        for widths, gap, h, upper, t, er, odd, even in coupons:
            with self.subTest(widths=widths, upper=upper, t=t):
                c, l, _ = section(widths, gap, h, upper, t, er, 0, 0, 1, 0, 24, 18)
                modes = modal(c, l)
                vals = (
                    [modes["single"]["z_ohm"]]
                    if even is None
                    else [modes["diff"]["z_ohm"] / 2, modes["common"]["z_ohm"] * 2]
                )
                for value, ref in zip(vals, (odd, even)):
                    self.assertLess(abs(value / ref - 1), 0.01)
                if upper:
                    for m in modes.values():
                        self.assertAlmostEqual(
                            C0 / m["velocity_m_s"], math.sqrt(er), places=8
                        )

    def test_route_arc_barrel_and_stub(self):
        layers = ["Upper", "Middle", "Lower"]
        depth = dict(zip(layers, [0, 0.3, 1.5]))
        tracks = [
            dict(
                a=[0, 0],
                mid=[1, 1],
                b=[2, 0],
                length_mm=math.pi,
                width_mm=0.2,
                layer="Upper",
            ),
            dict(a=[2, 0], b=[5, 0], length_mm=3, width_mm=0.2, layer="Middle"),
        ]
        net = dict(tracks=tracks, vias=[dict(xy=[2, 0], layers=layers, drill_mm=0.25)])
        edges = route(dict(nets={"A": net}), {"depth_mm": depth}, "A")
        self.assertAlmostEqual(
            sum(e["length_mm"] for e in edges), math.pi + 3.3, places=8
        )
        self.assertEqual(edges[1]["stubs_mm"], [0, 1.2])
        np.testing.assert_allclose(point(edges[0], 0.5), [1, 1], atol=1e-12)
        self.assertEqual(
            detect(["A_P", "A_N", "B+", "B-", "C_P"]), [["A_P", "A_N"], ["B+", "B-"]]
        )

    def test_matched_line_loss_delay_and_passivity(self):
        speed = 1.7e8
        z = 50
        frequency = 8e8
        length = 0.03
        c = np.array([[1 / (z * speed)]])
        seg = dict(
            C=c,
            L=np.array([[z / speed]]),
            loss=c * 0.014,
            widths=[0.34],
            lengths=[length * 1000],
            heights=[0.2],
            thickness=[0.035],
            speeds=[speed],
            stubs=[[]],
            stub_z=[50],
        )
        s, _, _ = response([seg], frequency, 50, 1.724e-8, 1)
        self.assertLessEqual(np.linalg.svd(s)[1].max(), 1 + 1e-12)
        self.assertLess(abs(s[0, 1] - s[1, 0]), 1e-10)
        self.assertLess(
            abs(np.angle(s[1, 0]) + 2 * math.pi * frequency * length / speed), 0.025
        )
        double, _, _ = response([seg, seg], frequency, 50, 1.724e-8, 1)
        self.assertLess(abs(abs(double[1, 0]) - abs(s[1, 0]) ** 2), 0.001)

    def test_via_in_landing_pad_contributes_the_full_barrel(self):
        layers = ["Top", "Back"]
        pads = {
            name: [dict(xy=xy, layer=layer)]
            for name, xy, layer in [("Near", [0, 0], "Top"), ("Far", [4, 0], "Back")]
        }
        net = dict(
            pads=pads,
            tracks=[dict(a=[0, 0], b=[4, 0], layer="Back", length_mm=4, width_mm=0.2)],
            vias=[dict(xy=[0, 0], layers=layers, drill_mm=0.25)],
        )
        for endpoints in (None, ["Near", "Far"]):
            path = route(
                {"nets": {"Trace": net}},
                {"depth_mm": dict(zip(layers, [0, 1.6]))},
                "Trace",
                endpoints,
            )
            self.assertAlmostEqual(sum(e["length_mm"] for e in path), 5.6)

    def test_pair_jog_and_slot_are_localized(self):
        def net(points):
            return dict(
                tracks=[
                    dict(
                        a=a, b=b, length_mm=math.dist(a, b), width_mm=0.28, layer="Top"
                    )
                    for a, b in zip(points, points[1:])
                ],
                pads={},
                vias=[],
            )

        def rect(x0, x1):
            return {"shell": [[x0, -10], [x1, -10], [x1, 10], [x0, 10]], "holes": []}

        g = {
            "layers": ["Top", "Base"],
            "nets": {
                "Lane_P": net([[0, 0], [5, 0], [6, -1], [7, 0], [12, 0]]),
                "Lane_N": net([[0, 0.43], [12, 0.43]]),
                "Foil": {"copper": {"Base": [rect(-1, 8.1), rect(8.24, 14)]}},
            },
        }
        mat = {
            "depth_mm": {"Top": 0, "Base": 0.18},
            "copper_mm": {"Top": 0.018, "Base": 0.018},
        }
        spec = dict(
            reference="Foil",
            detect_pairs=True,
            frequency_hz=[1e8],
            cells=10,
            margin=8,
            step_mm=0.8,
            match=[["Lane_P", "Lane_N"]],
            stack={"Top": {"plane": "Base", "er": 3.8}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            (r,), matches = transmission.studies(g, mat, spec, Path(tmp))
            self.assertTrue(all((Path(tmp) / f).exists() for f in r["files"]))
        self.assertAlmostEqual(r["skew_length_mm"], 2 * (math.sqrt(2) - 1))
        self.assertAlmostEqual(
            matches[0]["stations"][-1]["length_difference_mm"], r["skew_length_mm"]
        )
        gaps = [v for v in r["sections"] if not v["reference_resolved"]]
        self.assertTrue(gaps)
        self.assertAlmostEqual(sum(v["increment_mm"][1] for v in gaps), 0.14, places=4)
        self.assertLess(
            max(abs(v["skew_ps"]) for v in r["sections"] if v["xy"][0][0] < 4.5), 0.001
        )
        self.assertTrue(any("uncoupled" in v["flags"] for v in r["sections"]))
        g["nets"]["Foil"]["copper"]["Base"] = [
            {"shell": [[-1, -3], [14, -3], [14, 0.2], [-1, 0.2]], "holes": []}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            (r,), _ = transmission.studies(g, mat, spec, Path(tmp))
            gap_map = next(m for m in r["maps"] if "Missing reference" in m["name"])
            self.assertTrue(all(abs(p[1] - 0.43) < 1e-9 for p in gap_map["points"]))
            delay = next(m for m in r["maps"] if m["name"] == "Lane_N / delay")
            self.assertEqual(delay["points"][-1][:2], [12, 0.43])
            g["nets"]["Foil"]["copper"]["Base"] = []
            (r,), _ = transmission.studies(g, mat, spec, Path(tmp))
            self.assertTrue(all(s["local_z_ohm"] is None for s in r["sections"]))


if __name__ == "__main__":
    unittest.main()
