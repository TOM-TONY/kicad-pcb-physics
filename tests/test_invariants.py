"""Closed currents, translation symmetry and common conductor equivalence."""

import unittest
import numpy as np
from kicad_pcb_physics import mesh, pdn, studies


def rect(x0, y0, x1, y1):
    return {"shell": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": []}


def metal(polygons):
    return {"copper": {"Plane": polygons}, "pads": {}, "vias": [], "tracks": []}


def disk(x, y, current, radius):
    return {
        "current_A": current,
        "contacts": [{"layer": "Plane", "xy": [x, y], "radius_mm": radius}],
    }


material = {
    "depth_mm": {"Plane": 0},
    "copper_mm": {"Plane": 0.035},
    "rho_ohm_m": 1.72e-8,
    "plating_mm": 0.025,
}


class Invariants(unittest.TestCase):
    def test_floating_induced_current(self):
        outer = rect(0, 0, 3, 3)
        outer["holes"] = [rect(1, 1, 2, 2)["shell"]]
        sheet = mesh.Sheet(
            {"layers": ["Plane"], "nets": {"Trace": metal([outer, rect(5, 0, 8, 1)])}},
            material,
            "Trace",
            1,
        )
        xyz = sheet.positions_m()
        a, b = xyz[sheet.a], xyz[sheet.b]
        f = 1e5
        flux = 1e-6
        emf = -1j * np.pi * f * flux * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        _, _, flow, _, residual = sheet.solve(
            [disk(5.5, 0.5, 1, 0.2), disk(7.5, 0.5, -1, 0.2)], f, emf
        )
        reference = (
            2
            * np.pi
            * f
            * flux
            * 4e-6
            / abs(8 * mesh.slab(f, 35e-6, material["rho_ohm_m"]))
        )
        self.assertAlmostEqual(
            max(abs(flow[a[:, 0] < 0.004])) / reference, 1, places=10
        )
        self.assertLess(residual, 1e-10)

    def test_magnetic_translation(self):
        values = []
        for y in (0.5, 0.9):
            g = {
                "layers": ["Plane"],
                "nets": {"Trace": metal([rect(0, y - 0.1, 10, y + 0.1)])},
            }
            spec = {
                "name": "axis",
                "net": "Trace",
                "mesh_mm": 0.1,
                "frequency_Hz": [1e6],
                "ports": [disk(0.25, y, 1, 0.0801), disk(9.75, y, -1, 0.0801)],
                "field": {"frequency_Hz": 1e6, "x_mm": [10.75], "y_mm": [y], "z_mm": 0},
            }
            values.append(
                studies.electrical(g, material, spec)["near_field"]["max_B_T"]
            )
        self.assertLess(max(values), 1e-14)

    def test_shared_resistance_and_inductance(self):
        geo = {
            "layers": ["Plane"],
            "nets": {},
            "parts": {"Probe": {"layer": "Plane"}, "Bank": {"layer": "Plane"}},
        }
        for name, y, pin in (("West", 0.5, "a"), ("East", 2.5, "b")):
            n = metal([rect(0, y - 0.25, 10, y + 0.25)])
            for ref, x in (("Probe", 0.5), ("Bank", 9.5)):
                n["pads"][ref + "." + pin] = [
                    {
                        "layer": "Plane",
                        "xy": [x, y],
                        "polygons": [rect(x - 0.15, y - 0.25, x + 0.15, y + 0.25)],
                    }
                ]
            geo["nets"][name] = n
        cap = {
            "name": "small",
            "legs": [["Probe.a", "Bank.a"], ["Bank.b", "Probe.b"]],
            "capacitance_F": 10e-6,
            "esr_ohm": 0.002,
            "esl_H": 1e-9,
        }
        common = {
            "name": "Overlap",
            "voltage_V": 1,
            "frequency_Hz": [1e5, 2e6, 5e7],
            "feed": {"resistance_ohm": 1e6, "inductance_H": 0},
            "routing": {"pitch_mm": 0.1},
        }
        double = pdn.analyze(
            geo, material, {**common, "capacitors": [cap, {**cap, "name": "other"}]}
        )
        single = pdn.analyze(
            geo,
            material,
            {
                **common,
                "capacitors": [
                    {**cap, "capacitance_F": 20e-6, "esr_ohm": 0.001, "esl_H": 0.5e-9}
                ],
            },
        )
        self.assertTrue(
            np.allclose(double["z_abs_ohm"], single["z_abs_ohm"], rtol=1e-9, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(double["phase_deg"], single["phase_deg"], rtol=1e-9, atol=1e-9)
        )
        R = np.array(double["copper_transfer_ohm"])
        self.assertAlmostEqual(R[0, 0], R[0, 1], places=11)
