"""Reproducible manufactured solutions. Coordinates and fixtures are synthetic."""

import json
import math
import unittest
import numpy as np
from kicad_pcb_physics import fields, mesh, paths, pdn

MATERIAL = {
    "copper_mm": {"Top": 0.035, "Base": 0.035},
    "depth_mm": {"Top": 0, "Base": 1.4},
    "rho_ohm_m": 1.72e-8,
    "plating_mm": 0.025,
}


def rectangle(x0, y0, x1, y1):
    return {"shell": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "holes": []}


def strip():
    layers = ["Top", "Base"]
    net = {
        "pads": {},
        "vias": [],
        "copper": {"Top": [rectangle(0, 0, 10, 1)], "Base": []},
    }
    geo = {
        "layers": layers,
        "nets": {"Strip": net},
        "outline": [rectangle(0, 0, 10, 1)],
        "parts": {},
    }
    for name, x, i in (("entry", 0.3, 1), ("exit", 9.7, -1)):
        net["pads"][name] = [
            {
                "layer": "Top",
                "xy": [x, 0.5],
                "polygons": [rectangle(x - 0.08, -0.001, x + 0.08, 1.001)],
            }
        ]
    return geo, [
        {"contacts": ["entry"], "current_A": 1},
        {"contacts": ["exit"], "current_A": -1},
    ]


def checks():
    records = []

    def record(name, model, reference, tol):
        error = float(abs(model - reference) / abs(reference))
        records.append(
            {
                "name": name,
                "calculated": float(model),
                "reference": float(reference),
                "relative_error": error,
                "tolerance": tol,
            }
        )

    geo, ports = strip()
    s = mesh.Sheet(geo, MATERIAL, "Strip", 0.2)
    r = MATERIAL["rho_ohm_m"] * 0.0094 / (0.001 * 0.000035)
    _, term, current, watts, kcl = s.solve(ports)
    record("Copper strip potential drop", term[0], r, 0.005)
    record("Copper strip dissipated power", watts.sum(), r, 0.005)
    # Independent scalar diffusion FD through the thickness. Both faces see the same longitudinal E.
    rho = MATERIAL["rho_ohm_m"]
    t = 35e-6
    f = 40e6
    z = mesh.slab(f, t, rho)
    from scipy.linalg import solve_banded

    n = 301
    dz = t / (n - 1)
    u = 1j * 2 * np.pi * f * mesh.MU / rho * dz**2
    A = np.zeros((3, n - 2), complex)
    A[1] = -2 - u
    A[2, :-1] = 1
    A[0, 1:] = 1
    rhs = np.zeros(n - 2, complex)
    rhs[0] = rhs[-1] = -1
    E = np.r_[1, solve_banded((1, 1), A, rhs), 1]
    integral = dz * (E.sum() - (E[0] + E[-1]) / 2)
    record("Skin-effect AC sheet resistance", z.real, (rho / integral).real, 0.001)
    # Parallel plates with uniform cross-section and uniform injection; include both internal and gap inductance.
    _, v, _, _, _ = s.solve(ports, f, reference_height_mm=0.16)
    expected = (rho / integral + 2j * np.pi * f * mesh.MU * 0.00016) * 9.4
    record("AC reference-plane longitudinal drop", abs(v[0]), abs(expected), 0.005)
    p0 = [[-0.012, 0, 0], [0.012, 0, 0]]
    a, b, i = fields.filaments(p0, 1.0, 0.00008)
    offset = 0.003
    av, B = fields.potentials([[0, offset, 0]], a, b, i)
    proper = mesh.MU / (2 * np.pi * offset) * 0.012 / np.hypot(0.012, offset)
    record("Finite straight conductor magnetic field", B[0, 2], proper, 0.001)
    # Independently parallel two unequal foil levels into common terminals; impose a tangential EMF on one.
    from copy import deepcopy

    twin = deepcopy(geo)
    material = deepcopy(MATERIAL)
    material["copper_mm"]["Base"] = 0.0175
    metal = twin["nets"]["Strip"]
    metal["copper"]["Base"] = deepcopy(metal["copper"]["Top"])
    for pads in metal["pads"].values():
        pads.append({**pads[0], "layer": "Base"})
    sheet = mesh.Sheet(twin, material, "Strip", 0.2)
    zz = sheet.positions_m()
    bottom = sheet.levels[sheet.a] == "Base"
    source = 1j * 0.4 * (zz[sheet.b, 0] - zz[sheet.a, 0]) * bottom
    _, _, flow, _, _ = sheet.solve(ports, f, source, 0.1)
    cut = (zz[sheet.a, 0] < 0.005) & (zz[sheet.b, 0] > 0.005)
    topz = (mesh.slab(f, 35e-6, rho) + 2j * np.pi * f * mesh.MU * 0.0001) * 9.4
    basez = (mesh.slab(f, 17.5e-6, rho) + 2j * np.pi * f * mesh.MU * 0.0001) * 9.4
    expected = (topz + 1j * 0.4 * 0.0094) / (topz + basez)
    record(
        "Driven return-current split",
        abs(flow[cut & bottom].sum()),
        abs(expected),
        0.001,
    )
    # Analytic vector potential sets inductive longitudinal electric field.
    analytic_A = mesh.MU / (2 * np.pi) * np.arcsinh(0.012 / offset)
    record("Finite conductor vector potential", av[0, 0], analytic_A, 0.001)
    q = 1e-12
    height = 0.002
    zobs = 0.006
    electric = fields.electric(
        np.array([[0.0, 0, zobs]]), [[0, 0, height]], np.array([q]), 0
    )
    exact = (
        q
        / (4 * np.pi * fields.EPS)
        * (1 / (zobs - height) ** 2 - 1 / (zobs + height) ** 2)
    )
    record("Electrostatic image-charge field", electric[0, 2], exact, 1e-10)
    points = np.array(
        [[0, 0, 0], [0.01, 0, 0], [0.01, 0.02, 0], [0, 0.02, 0], [0, 0, 0]]
    )
    recv = fields.receive(points, [0, 0, 0], [0, 0, 2e-6], 1e6)
    record(
        "Faraday receiving-loop voltage",
        abs(recv),
        2 * np.pi * 1e6 * 0.01 * 0.02 * 2e-6,
        1e-12,
    )
    dip = fields.dipole_limit(points, 1, 1e8, 10)
    # Broadside far field from independent retarded filament integration.
    a, b, i = fields.filaments(points, 1.0, 0.0001)
    k = 2 * np.pi * 1e8 / fields.LIGHT
    integ = ((b - a) * np.exp(1j * k * (a[:, 0] + b[:, 0]) / 2)[:, None]).sum(axis=0)
    exact = 2 * np.pi * 1e8 * mesh.MU / (4 * np.pi * 10) * abs(integ[1])
    record("Small-loop broadside far field", dip["maximum_far_E_V_m"], exact, 0.001)
    e = fields.distributed_emission(a, b, i, 1e8, 10)
    record("Distributed-current dipole field", e["far_E_bound_V_m"], exact, 0.001)
    loop = np.array([[0, 0, 0], [10, 0, 0], [10, 20, 0], [0, 20, 0], [0, 0, 0]])
    record(
        "Rectangular loop inductance",
        paths.neumann(loop, 0.16, 0.12),
        paths.rectangle(10, 20, 0.16),
        0.005,
    )
    R = 0.05
    L = 8e-9
    C = 2e-6
    resonant = 1 / (2 * np.pi * math.sqrt(L * C))
    feed = {"resistance_ohm": 1e6, "inductance_H": 0.0001}
    zz = pdn.impedance([resonant], [C], [R], [[L]], feed)[0]
    record("Series-resonant capacitor port", float(abs(zz)), R, 1e-6)
    # Loaded receiver between two synthetic pads, with a reference plane and independent series source.
    rec = geo.copy()
    rec["parts"] = {"entry": {"layer": "Top"}, "exit": {"layer": "Top"}}
    rec["nets"]["Strip"]["tracks"] = [
        {"a": [0.3, 0.5], "b": [9.7, 0.5], "layer": "Top"}
    ]
    loaded = fields.exposure(
        rec,
        MATERIAL,
        {
            "name": "Coupon pickup",
            "frequency_Hz": 8e6,
            "E_V_m": [3, 0, 0],
            "B_T": [0, 2e-6, 0],
            "reference_depth_mm": 1.4,
            "reference_xy_mm": [0.3, 0.5],
            "default_termination": {
                "harness_V": [0.01, 0],
                "source_ohm": [50, 0],
                "load_ohm": [50, 0],
            },
        },
    )
    v = next(v["terminal_V"] for v in loaded["terminal_emf"] if v["pin"] == "exit")
    record(
        "Loaded pickup voltage",
        abs(complex(*v)),
        abs((0.01 + 3 * 0.0094 + 2j * np.pi * 8e6 * 2e-6 * 0.0094 * 0.0014) / 2),
        1e-10,
    )
    return records


class Validation(unittest.TestCase):
    def test_manufactured_solutions(self):
        for row in checks():
            with self.subTest(case=row["name"]):
                self.assertLessEqual(row["relative_error"], row["tolerance"])

    def test_bias_curve_limits(self):
        curve = {"capacitance_F": 2e-6, "bias_curve": [[0, 1], [4, 0.6]]}
        self.assertAlmostEqual(pdn.biased(curve, 2), 1.6e-6, places=13)
        with self.assertRaises(ValueError):
            pdn.biased(curve, 5)

    def test_overlapping_ports(self):
        geo, _ = strip()
        s = mesh.Sheet(geo, MATERIAL, "Strip", 0.2)
        with self.assertRaises(ValueError):
            s.solve(
                [
                    {"contacts": ["entry"], "current_A": 1},
                    {"contacts": ["entry"], "current_A": -1},
                ]
            )


if __name__ == "__main__":
    rows = checks()
    print(json.dumps(rows, indent=2))
    if any(r["relative_error"] > r["tolerance"] for r in rows):
        raise SystemExit(1)
