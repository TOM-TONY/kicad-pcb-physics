"""Geometry-bound electrical studies and field-map serialization."""

from collections import defaultdict
import numpy as np
from . import fields, paths
from .mesh import Sheet


def bins(layer, xy, values, pitch=1.0, kind="mean"):
    rows = {}
    for l, (x, y), value in zip(layer, xy, values):
        key = (l, int(np.floor(x / pitch)), int(np.floor(y / pitch)))
        if key not in rows:
            rows[key] = [0.0, 0]
        rows[key][0] += float(value)
        rows[key][1] += 1
    output = defaultdict(list)
    for (l, i, j), (v, n) in rows.items():
        output[l].append(
            [(i + 0.5) * pitch, (j + 0.5) * pitch, v if kind == "sum" else v / n]
        )
    return output


def mapset(name, unit, layer, xy, values, pitch, kind="mean"):
    return [
        {"name": name, "unit": unit, "layer": l, "points": pts}
        for l, pts in bins(layer, xy, values, pitch, kind).items()
    ]


def electrical(geo, mat, case):
    sheet = Sheet(geo, mat, case["net"], case["mesh_mm"], case.get("bounds_mm"))
    freqs = case.get("frequency_Hz", [0])
    maps = []
    rows = []
    heats = []
    sample = {}
    ports = case["ports"]
    amp = sum(p["current_A"] for p in ports if p["current_A"] > 0)
    xyz = sheet.positions_m()
    delta = xyz[sheet.b] - xyz[sheet.a]
    mid = (xyz[sheet.a] + xyz[sheet.b]) / 2
    l = sheet.levels[sheet.a]
    view = case.get("map_pitch_mm", 1)
    source = None
    drive_path = case.get("exciting_path_mm")
    if "exciting_legs" in case:
        legs = case["exciting_legs"]
        cfg = {
            "pitch_mm": 0.1,
            "roi_margin_mm": 8,
            "centerline_penalty_mm": 0.04,
            "width_cap_mm": 2.0,
            **case.get("routing", {}),
        }
        prepared = paths.prepare(geo, mat, legs)
        drive_path = np.vstack(
            [paths.route(prepared, *pair, cfg["pitch_mm"], cfg)[0] for pair in legs]
        )
    if drive_path is not None:
        a, b, i = fields.filaments(
            np.array(drive_path) * 0.001, case.get("exciting_A", 1.0), 0.0004
        )
        vector, _ = fields.potentials(
            mid, a, b, i, case.get("wire_radius_mm", 0.08) * 0.001
        )
        source = np.sum(vector * delta, axis=1)
    for f in freqs:
        v, u, I, w, kcl = sheet.solve(
            ports,
            f,
            None if source is None or not f else -2j * np.pi * f * source,
            case.get("return_spacing_mm", 0),
        )
        dv = u[0] - u[-1]
        rows.append(
            {
                "frequency_Hz": f,
                "resistance_ohm": float(dv.real / amp),
                "reactance_ohm": float(dv.imag / amp),
                "drop_V": float(abs(dv)),
                "copper_W": float(w.sum()),
                "kcl_error_A": kcl,
                "terminal_V": [[float(z.real), float(z.imag)] for z in u],
            }
        )
        active = abs(I) > 1e-15
        label = case["name"] + " " + str(f) + " Hz"
        via = np.array([e[5] for e in sheet.edges]) & active
        rows[-1]["via_current"] = [
            {
                "xy_mm": pt.tolist(),
                "from": str(sheet.levels[sheet.a[k]]),
                "to": str(sheet.levels[sheet.b[k]]),
                "amperes": [float(I[k].real), float(I[k].imag)],
            }
            for k, pt in zip(np.flatnonzero(via), mid[via, :2] * 1000)
        ]
        if via.any():
            maps += mapset(
                label + " barrel current",
                "A",
                l[via],
                mid[via, :2] * 1000,
                abs(I[via]),
                view,
            )
        if drive_path is not None and active.any():
            from scipy.spatial import cKDTree

            tree = cKDTree((a[:, :2] + b[:, :2]) / 2)
            distance = tree.query(mid[active, :2])[0] * 1000
            weight = abs(I[active]) * sheet.length[active]
            rows[-1]["rms_return_spread_mm"] = float(
                np.sqrt(np.average(distance**2, weights=weight))
            )
        maps += mapset(
            label + " potential", "V", sheet.levels, sheet.xy, abs(v) if f else v, view
        )
        maps += mapset(
            label + " current density",
            "A/mm2",
            l[active],
            mid[active, :2] * 1000,
            abs(I[active]) / (sheet.width[active] * sheet.thick[active]),
            view,
        )
        maps += mapset(
            label + " copper loss",
            "W",
            l[active],
            mid[active, :2] * 1000,
            w[active],
            view,
            "sum",
        )
        if f == 0:
            for layer, pts in bins(
                l[active], mid[active, :2] * 1000, w[active], view, "sum"
            ).items():
                heats.extend(
                    {"layer": layer, "x_mm": x, "y_mm": y, "watts": power}
                    for x, y, power in pts
                )
        if (
            source is not None
            and f
            and "radiation_distance_m" in case
            and len(ports) == 2
        ):
            # Close the physical loop through the component terminal centroids. Finite
            # contacts split current over cells; retain those separate closure segments.
            if not np.isclose(case.get("exciting_A", 1.0), ports[0]["current_A"]):
                raise ValueError(
                    "Distributed radiation requires matching drive and return current"
                )
            div = np.zeros(len(v), complex)
            np.add.at(div, sheet.a, I)
            np.add.at(div, sheet.b, -I)
            starts = [xyz[sheet.a[active]], a]
            ends = [xyz[sheet.b[active]], b]
            amps = [I[active], i]
            for pin, anchor in zip(ports, [b[-1], a[0]]):
                ns = np.unique(
                    [n for con in pin["contacts"] for n in sheet.electrode(con)]
                )
                starts.append(np.broadcast_to(anchor, (len(ns), 3)))
                ends.append(xyz[ns])
                amps.append(div[ns])
            rad = fields.distributed_emission(
                np.vstack(starts),
                np.vstack(ends),
                np.concatenate(amps),
                f,
                case["radiation_distance_m"],
            )
            rows[-1]["distributed_emission"] = rad
            if rad["applicable"]:
                maps += mapset(
                    label + " radiating current",
                    "A",
                    l[active],
                    mid[active, :2] * 1000,
                    abs(I[active]),
                    view,
                )
        if "field" in case and f == case["field"]["frequency_Hz"]:
            s = case["field"]
            gx, gy = np.meshgrid(s["x_mm"], s["y_mm"])
            obs = (
                np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, s["z_mm"])))
                * 0.001
            )
            # Keep each sheet current at its solved location, including isolated eddy currents.
            A, B = fields.potentials(
                obs,
                xyz[sheet.a[active]],
                xyz[sheet.b[active]],
                I[active],
                s.get("source_radius_mm", 0.01) * 0.001,
            )
            if source is not None:
                aa, bb = fields.potentials(
                    obs, a, b, i, case.get("wire_radius_mm", 0.08) * 0.001
                )
                A += aa
                B += bb
            E = -2j * np.pi * f * A
            if "sheet_charge" in s:
                cap = s["sheet_charge"]
                plane = cap["image_z_mm"] * 0.001
                inplane = np.array([not e[5] for e in sheet.edges])
                area = np.zeros(len(v))
                face = sheet.length[inplane] * sheet.width[inplane] * 0.25e-6
                np.add.at(area, sheet.a[inplane], face)
                np.add.at(area, sheet.b[inplane], face)
                distance = np.abs(xyz[:, 2] - plane)
                eligible = (area > 0) & (distance > 1e-7)
                charge = (
                    fields.EPS
                    * cap["epsilon_r"]
                    * area[eligible]
                    * (v[eligible] + cap.get("common_mode_V", 0))
                    / distance[eligible]
                )
                E += fields.electric(obs, xyz[eligible], charge, plane)
            if "charge_nodes" in s:
                charges = s["charge_nodes"]
                locations = []
                for c in charges:
                    if "pin" in c:
                        pins = geo["nets"][c["net"]]["pads"][c["pin"]]
                        side = geo["parts"][c["pin"].rsplit(".", 1)[0]]["layer"]
                        pad = next(p for p in pins if p["layer"] == side)
                        locations.append([*pad["xy"], mat["depth_mm"][side]])
                    else:
                        locations.append(c["xyz_mm"])
                E += fields.electric(
                    obs,
                    np.array(locations) * 0.001,
                    np.array([c["cap_F"] * c["voltage_V"] for c in charges]),
                    s["image_z_mm"] * 0.001,
                )
            layers = [s.get("display_layer", geo["layers"][0])] * len(obs)
            maps += mapset(
                label + " magnetic flux density",
                "T",
                layers,
                obs[:, :2] * 1000,
                np.linalg.norm(B, axis=1),
                view,
            )
            maps += mapset(
                label + " electric field",
                "V/m",
                layers,
                obs[:, :2] * 1000,
                np.linalg.norm(E, axis=1),
                view,
            )
            sample = {
                "max_B_T": float(max(np.linalg.norm(B, axis=1))),
                "max_E_V_m": float(max(np.linalg.norm(E, axis=1))),
            }
    return {
        "name": case["name"],
        "net": case["net"],
        "rows": rows,
        "maps": maps,
        "heat_cells": heats,
        "near_field": sample,
    }


def reception(geo, mat, spec):
    name = spec["name"]
    maps = []
    rows = []
    for item in spec["paths"]:
        a, b = item["legs"][0]
        all_legs = item["legs"]
        g = paths.prepare(geo, mat, all_legs)
        cfg = {
            "roi_margin_mm": 8,
            "centerline_penalty_mm": 0.04,
            "width_cap_mm": 2.0,
            "pitch_mm": 0.12,
            **spec.get("routing", {}),
        }
        legs = [paths.route(g, *leg, cfg["pitch_mm"], cfg)[0] for leg in all_legs]
        p = np.vstack(legs) * 0.001
        # The electrostatic term uses the measuring electrodes, irrespective of the specified magnetic return.
        e = np.array(spec["E_V_m"])
        bfield = np.array(spec["B_T"])
        induced = fields.receive(p, [0, 0, 0], bfield, spec["frequency_Hz"]) + np.dot(
            e, (legs[0][-1] - legs[0][0]) * 0.001
        )
        harness = complex(*item.get("harness_V", [0, 0]))
        zsrc = complex(*item.get("source_ohm", [0, 0]))
        zload = complex(*item.get("load_ohm", [1, 0]))
        voltage = (induced + harness) * zload / (zsrc + zload)
        rows.append(
            {
                "name": item["name"],
                "induced_V": abs(induced),
                "terminal_V": abs(voltage),
            }
        )
        for idx, layer in enumerate(geo["layers"]):
            coords = [
                [*pt[:2] * 1000, float(abs(voltage))]
                for pt in p
                if abs(pt[2] * 1000 - g["z"][idx]) < 1e-7
            ]
            if coords:
                maps.append(
                    {
                        "name": name + " / " + item["name"],
                        "layer": layer,
                        "unit": "V",
                        "points": coords,
                    }
                )
    return {"name": name, "rows": rows, "maps": maps}
