import unittest
from kicad_pcb_physics.stitching import survey, topology


def rect(x, y, w, h):
    return {"shell": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], "holes": []}


class PlacementGeometry(unittest.TestCase):
    def test_hole_edge_pad_other_copper_and_keepout(self):
        metal = {l: [rect(0, 0, 9, 7)] for l in ["Upper", "Lower"]}
        small = rect(2.8, 2.8, 0.4, 0.4)
        g = dict(
            layers=list(metal),
            outline=[rect(0, 0, 9, 7)],
            holes=[small],
            pad_obstacles=[{"layer": "Upper", "polygons": [rect(3.8, 2.8, 0.4, 0.4)]}],
            extra_metal={},
            via_keepouts=[
                {"layers": ["Lower"], "polygons": [rect(1.8, 0.8, 0.4, 0.4)]}
            ],
            nets={
                "Foil": {"copper": metal, "vias": []},
                "Signal": {"copper": {"Lower": [rect(5.8, 2.8, 0.4, 0.4)]}, "vias": []},
            },
        )
        spec = dict(
            net="Foil",
            land_mm=0.6,
            drill_mm=0.3,
            clearance_mm=0.2,
            hole_clearance_mm=0.25,
            edge_clearance_mm=0.35,
            search_step_mm=1,
            via_spacing_mm=1,
        )
        _, sites = survey(g, spec)
        for blocked in [[3, 3], [4, 3], [6, 3], [2, 1], [0, 3]]:
            self.assertNotIn(blocked, sites)
        self.assertIn([7, 5], sites)

    def test_single_anchor_and_neck(self):
        copper = [
            rect(0, 0, 3, 3),
            rect(3, 1.4, 2, 0.2),
            rect(5, 0, 3, 3),
            rect(10, 0, 2, 2),
        ]
        via = dict(xy=[11, 1], layers=["Top", "Back"], drill_mm=0.25, land_mm=0.55)
        g = dict(
            nets={
                "Foil": {
                    "copper": {"Top": copper, "Back": [rect(10, 0, 2, 2)]},
                    "vias": [via],
                }
            },
            holes=[],
        )
        rows, _ = topology(g, {"net": "Foil", "neck_mm": 0.5})
        self.assertEqual(len([r for r in rows if r["kind"] == "neck-lobe"]), 2)
        self.assertTrue(
            any(r["ground_contacts"] == 1 and r["layer"] == "Top" for r in rows)
        )

    def test_a_well_stitched_group_can_still_be_floating(self):
        foils = {
            "Top": [rect(0, 0, 6, 2), rect(7, 0, 3, 2)],
            "Back": [rect(0, 0, 4, 2), rect(4.5, 0, 5.5, 2)],
        }
        vias = [dict(xy=[x, 1], layers=list(foils), drill_mm=0.3) for x in (1, 3, 8, 9)]
        g = {"nets": {"Foil": {"copper": foils, "vias": vias}}, "holes": []}
        rows, _ = topology(g, {"net": "Foil"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(
                r["ground_contacts"] == 2 and not r["connected_to_largest_patch"]
                for r in rows
            )
        )
