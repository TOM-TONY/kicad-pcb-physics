"""Quasistatic filament fields, receiving loops and magnetically driven copper returns."""

import numpy as np
from shapely.geometry import LineString
from .mesh import MU, polygon

EPS = 8.8541878128e-12
LIGHT = 299792458.0


def filaments(points, current, step_m):
    points = np.asarray(points, dtype=float)
    starts = []
    ends = []
    for a, b in zip(points, points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        sub = np.linspace(a, b, n + 1)
        starts.extend(sub[:-1])
        ends.extend(sub[1:])
    return np.asarray(starts), np.asarray(ends), np.full(len(starts), current)


def potentials(obs, start, end, amps, radius_m=0):
    """Integrate finite straight segments with optional finite-radius regularisation, in SI."""
    obs = np.asarray(obs, float)
    starts = np.asarray(start, float)
    seg = np.asarray(end, float) - starts
    lengths = np.linalg.norm(seg, axis=1)
    valid = lengths > 1e-15
    starts = starts[valid]
    lengths = lengths[valid]
    amps = np.asarray(amps)[valid]
    direction = seg[valid] / lengths[:, None]
    A = np.zeros((len(obs), 3), dtype=np.result_type(amps, float))
    B = np.zeros_like(A)
    for i in range(0, len(amps), 128):
        cut = slice(i, i + 128)
        unit = direction[cut]
        for j in range(0, len(obs), 256):
            q = slice(j, j + 256)
            r = obs[q, None, :] - starts[None, cut, :]
            t0 = np.einsum("ijk,jk->ij", r, unit)
            t1 = t0 - lengths[cut]
            perpendicular = r - t0[:, :, None] * unit
            rr = np.einsum("ijk,ijk->ij", perpendicular, perpendicular) + radius_m**2
            if np.any((rr < 1e-34) & (t0 >= 0) & (t1 <= 0)):
                raise ValueError("Observation on an ideal filament")
            rr = np.maximum(rr, 1e-34)
            rad = np.sqrt(rr)
            w = (np.arcsinh(t0 / rad) - np.arcsinh(t1 / rad)) * amps[cut]
            A[q] += MU / (4 * np.pi) * np.einsum("ij,jk->ik", w, unit)
            weight = (t0 / np.hypot(rad, t0) - t1 / np.hypot(rad, t1)) * amps[cut] / rr
            B[q] += (
                MU
                / (4 * np.pi)
                * np.einsum("ij,ijk->ik", weight, np.cross(unit, perpendicular))
            )
    return A, B


def electric(obs, points, charge, image_z_m):
    E = np.zeros((len(obs), 3), dtype=np.result_type(charge, float))
    for sign in (1, -1):
        p = np.array(points, copy=True)
        if sign < 0:
            p[:, 2] = 2 * image_z_m - p[:, 2]
        for i in range(0, len(p), 128):
            r = np.asarray(obs)[:, None, :] - p[None, i : i + 128, :]
            d = np.linalg.norm(r, axis=2)
            if np.any(d <= 0):
                raise ValueError("Observation on a charge node")
            E += (
                sign
                * np.sum(
                    charge[None, i : i + 128, None] * r / d[:, :, None] ** 3, axis=1
                )
                / (4 * np.pi * EPS)
            )
    return E


def receive(points, field_e, field_b, frequency):
    p = np.asarray(points)
    # Uniform fields. Close the magnetic integration path by the specified return or straight chord.
    area = 0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
    return np.dot(field_e, p[-1] - p[0]) - 2j * np.pi * frequency * np.dot(
        field_b, area
    )


def dipole_limit(points, amps, frequency, distance):
    p = np.asarray(points)
    if np.linalg.norm(p[0] - p[-1]) > 1e-9:
        raise ValueError("Magnetic dipole estimate requires a closed current loop")
    diameter = np.linalg.norm(np.ptp(p, axis=0))
    k = 2 * np.pi * frequency / LIGHT
    valid = k * diameter <= 0.3 and k * distance >= 3 and distance > 3 * diameter
    moment = 0.5 * amps * np.cross(p[:-1], p[1:]).sum(axis=0)
    return {
        "applicable": bool(valid),
        "kr": k * distance,
        "electrical_size": k * diameter,
        "maximum_far_E_V_m": float(
            376.730313668 * k * k * np.linalg.norm(moment) / (4 * np.pi * distance)
        )
        if valid
        else None,
    }


def return_gaps(geo, signal, plane, layer, ref_layer):
    mask = polygon(geo["nets"][plane]["copper"][ref_layer])
    out = []
    for segment in geo["nets"][signal]["tracks"]:
        if segment["layer"] != layer:
            continue
        if "mid" in segment:
            raise ValueError("Reference-gap distances require straight signal tracks")
        line = LineString([segment["a"], segment["b"]])
        gap = line.difference(mask)
        if gap.length > 1e-6:
            out.append(
                {
                    "a": segment["a"],
                    "b": segment["b"],
                    "missing_mm": gap.length,
                    "centre_mm": list(gap.interpolate(0.5, normalized=True).coords[0]),
                }
            )
    return out


def exposure(geo, material, spec):
    """Open-circuit contributions with a declared parallel reference plane and uniform incident fields."""
    maps = []
    tracks = []
    pads = []
    electric_v_m = np.array(spec["E_V_m"])
    magnetic_t = np.array(spec["B_T"])
    plane = spec["reference_depth_mm"] * 0.001
    origin = np.array(spec["reference_xy_mm"]) * 0.001
    frequency = spec["frequency_Hz"]
    selected = spec.get("nets") or geo["nets"].keys()

    def contribution(a, b, z):
        ends = np.asarray([a, b], dtype=float) * 0.001
        path = np.array(
            [[*ends[0], z], [*ends[1], z], [*ends[1], plane], [*ends[0], plane]]
        )
        return np.dot(electric_v_m[:2], ends[1] - ends[0]) + receive(
            path, [0, 0, 0], magnetic_t, frequency
        )

    for name in selected:
        net = geo["nets"][name]
        by_layer = {l: [] for l in geo["layers"]}
        for tr in net["tracks"]:
            z = material["depth_mm"][tr["layer"]] * 0.001
            # A projected return directly under an arc spans the same vertical vector area as its chord.
            v = contribution(tr["a"], tr["b"], z)
            tracks.append(
                {
                    "net": name,
                    "layer": tr["layer"],
                    "a": tr["a"],
                    "b": tr["b"],
                    "emf_V": [float(v.real), float(v.imag)],
                }
            )
            mid = (np.asarray(tr["a"]) + tr["b"]) / 2
            by_layer[tr["layer"]].append([*mid, float(abs(v))])
        for layer, pts in by_layer.items():
            if pts:
                maps.append(
                    {
                        "name": spec["name"] + " / " + name + " trace pickup",
                        "unit": "V",
                        "layer": layer,
                        "points": pts,
                    }
                )
        points = {l: [] for l in geo["layers"]}
        for pin, records in net["pads"].items():
            selected_pad = next(
                (
                    p
                    for p in records
                    if p["layer"] == geo["parts"][pin.rsplit(".", 1)[0]]["layer"]
                ),
                records[0],
            )
            layer = selected_pad["layer"]
            xy = selected_pad["xy"]
            v = contribution(origin * 1000, xy, material["depth_mm"][layer] * 0.001)
            term = spec.get("terminations", {}).get(
                pin, spec.get("default_termination", {})
            )
            vs = complex(*term.get("harness_V", [0, 0]))
            zs = complex(*term.get("source_ohm", [0, 0]))
            zl = complex(*term.get("load_ohm", [1, 0]))
            loaded = (v + vs) * zl / (zs + zl)
            pads.append(
                {
                    "pin": pin,
                    "net": name,
                    "layer": layer,
                    "xy": xy,
                    "emf_V": [float(v.real), float(v.imag)],
                    "terminal_V": [float(loaded.real), float(loaded.imag)],
                }
            )
            points[layer].append([*xy, float(abs(loaded))])
        for layer, pts in points.items():
            if pts:
                maps.append(
                    {
                        "name": spec["name"] + " / " + name + " pad pickup",
                        "unit": "V",
                        "layer": layer,
                        "points": pts,
                    }
                )
    return {
        "name": spec["name"],
        "trace_emf": tracks,
        "terminal_emf": pads,
        "maps": maps,
    }


def distributed_emission(starts, ends, currents, frequency, distance):
    """Magnetic dipole of a closed, electrically small distributed harmonic current."""
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    amps = np.asarray(currents)
    centre = (starts.min(axis=0) + ends.max(axis=0)) / 2
    dl = ends - starts
    closure = np.linalg.norm(np.sum(amps[:, None] * dl, axis=0))
    size = np.linalg.norm(np.ptp(np.vstack((starts, ends)), axis=0))
    k = 2 * np.pi * frequency / LIGHT
    moment = 0.5 * np.sum(
        amps[:, None] * np.cross(starts - centre, ends - centre), axis=0
    )
    valid = bool(
        closure <= 1e-7 * max(1e-12, np.sum(abs(amps) * np.linalg.norm(dl, axis=1)))
        and k * size <= 0.3
        and k * distance >= 3
        and distance > 3 * size
    )
    return {
        "applicable": valid,
        "closure_A_m": float(closure),
        "magnetic_moment_A_m2": float(np.linalg.norm(moment)),
        "far_E_bound_V_m": float(
            376.730313668 * k * k * np.linalg.norm(moment) / (4 * np.pi * distance)
        )
        if valid
        else None,
    }
