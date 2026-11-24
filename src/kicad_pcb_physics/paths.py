"""Quasistatic loop extraction from selected filled-metal paths."""

import math


def route(g, start, end, pitch, cfg):
    """Trace an eight-neighbour copper path and plated transitions, in millimetres."""
    import numpy as np
    import shapely as sh
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    a, b = [g["ports"][p] for p in (start, end)]
    net = a["net"]
    if net != b["net"]:
        raise ValueError((start, end, net, b["net"]))
    ends = np.array([a["xy"], b["xy"]])
    margin = cfg["roi_margin_mm"]
    lower = np.floor((ends.min(axis=0) - margin) / pitch) * pitch
    nx, ny = np.ceil((ends.max(axis=0) + margin - lower) / pitch).astype(int) + 1
    xx, yy = np.meshgrid(
        lower[0] + np.arange(nx) * pitch, lower[1] + np.arange(ny) * pitch
    )
    masks = []
    for geom in g["metal"][net]:
        sh.prepare(geom)
        masks.append(sh.contains_xy(geom, xx, yy))
    mask = np.asarray(masks)
    ids = np.full(mask.shape, -1, dtype=int)
    ids[mask] = np.arange(mask.sum())
    clearance = np.stack([distance_transform_edt(m) * pitch for m in mask])
    first, second, cost = [], [], []
    for dx, dy in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
        xs = slice(max(0, -dx), min(nx, nx - dx))
        ys = slice(0, ny - dy)
        xt = slice(max(0, dx), min(nx + dx, nx))
        yt = slice(dy, ny)
        i, j = ids[:, ys, xs], ids[:, yt, xt]
        valid = (i >= 0) & (j >= 0)
        if dx and dy:
            valid &= mask[:, yt, xs] | mask[:, ys, xt]
        rad = (clearance[:, ys, xs][valid] + clearance[:, yt, xt][valid]) / 2
        first.append(i[valid])
        second.append(j[valid])
        cost.append(
            pitch * math.hypot(dx, dy) * (1 + cfg["centerline_penalty_mm"] / rad)
        )

    def cell(xy):
        x, y = np.rint((xy - lower) / pitch).astype(int)
        return int(y), int(x)

    drills = {}
    for x, y, d, hits in g["barrels"].get(net, []):
        i, j = cell([x, y])
        if 0 <= i < ny and 0 <= j < nx:
            hits = [k for k in hits if mask[k, i, j]]
            for k, m in zip(hits, hits[1:]):
                u, v = ids[k, i, j], ids[m, i, j]
                first.append([u])
                second.append([v])
                cost.append([abs(g["z"][k] - g["z"][m])])
                drills[min(u, v), max(u, v)] = d
    graph = coo_matrix(
        (np.concatenate(cost), (np.concatenate(first), np.concatenate(second))),
        shape=(mask.sum(), mask.sum()),
    ).tocsr()

    def port_id(p):
        y, x = cell(p["xy"])
        return ids[p["layer"], y, x]

    u, v = port_id(a), port_id(b)
    if min(u, v) < 0:
        raise LookupError(f"{start}/{end} contact outside metal")
    dist, prev = dijkstra(graph, directed=False, indices=u, return_predecessors=True)
    if not np.isfinite(dist[v]):
        raise LookupError(f"{start}/{end} no resolved copper path")
    path = [v]
    while path[-1] != u:
        path.append(prev[path[-1]])
    path.reverse()
    indices = np.column_stack(np.nonzero(mask))[path]
    pts = np.array(
        [[lower[0] + x * pitch, lower[1] + y * pitch, g["z"][k]] for k, y, x in indices]
    )
    widths = []
    layers = []
    for idx, (k, y, x) in enumerate(indices[:-1]):
        m = indices[idx + 1, 0]
        if k != m:
            key = tuple(sorted(path[idx : idx + 2]))
            widths.append(drills[key])
            layers.append(-1)
        else:
            widths.append(
                min(cfg["width_cap_mm"], max(pitch, 2 * clearance[k, y, x] - pitch))
            )
            layers.append(int(k))
    # Restore physical contact centres after path finding.
    pts[0, :2] = a["xy"]
    pts[-1, :2] = b["xy"]
    return pts, np.asarray(widths), np.asarray(layers), net


def resample(points, step):
    import numpy as np

    arr = []
    for a, b in zip(points, points[1:]):
        length = np.linalg.norm(b - a)
        if length > 1e-9:
            arr.extend(
                np.linspace(a, b, max(1, math.ceil(length / step)), endpoint=False)
            )
    return np.vstack((arr, points[-1]))


def neumann(points, radius, step):
    """Regularised filament integral, with the analytic straight self term; nH."""
    import numpy as np
    from scipy.spatial.distance import cdist

    p = resample(points, step)
    d = np.diff(p, axis=0)
    mid = (p[:-1] + p[1:]) / 2
    length = np.linalg.norm(d, axis=1)
    s = 0.0
    for i in range(0, len(d), 128):
        block = d[i : i + 128]
        mutual = (block @ d.T) / np.sqrt(
            cdist(mid[i : i + 128], mid, "sqeuclidean") + radius**2
        )
        s += mutual.sum()
    exact = 2 * (
        length * np.arcsinh(length / radius) - np.hypot(length, radius) + radius
    )
    return float(0.1 * (s + sum(exact - length**2 / radius)))


def rectangle(a, b, r):
    def parallel(l, d):
        return 2 * (l * math.asinh(l / d) - math.hypot(l, d) + d)

    return 0.2 * (
        parallel(a, r)
        + parallel(b, r)
        - parallel(a, math.hypot(r, b))
        - parallel(b, math.hypot(r, a))
    )


def prepare(geo, material, legs):
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from .mesh import polygon

    pins = {pin for pair in legs for pin in pair}
    layers = geo["layers"]
    result = {
        "ports": {},
        "barrels": {},
        "metal": {},
        "z": [material["depth_mm"][layer] for layer in layers],
    }
    for name, net in geo["nets"].items():
        chosen = pins.intersection(net["pads"])
        if not chosen:
            continue
        lands = [[] for layer in layers]
        barrels = []
        for via in net["vias"]:
            indices = [layers.index(l) for l in via["layers"]]
            for i in indices:
                # A centreline enters the plated barrel at its rim; its axial link is at the centre.
                lands[i].append(Point(*via["xy"]).buffer(via["land_mm"] / 2))
            barrels.append([*via["xy"], via["drill_mm"], indices])
        result["barrels"][name] = barrels
        for p in chosen:
            ref = p.rsplit(".", 1)[0]
            layer = geo["parts"][ref]["layer"]
            pad = next(entry for entry in net["pads"][p] if entry["layer"] == layer)
            result["ports"][p] = {
                "xy": pad["xy"],
                "net": name,
                "layer": layers.index(layer),
            }
        result["metal"][name] = [
            unary_union([polygon(net["copper"][l]), *lands[i]])
            for i, l in enumerate(layers)
        ]
    if pins - set(result["ports"]):
        raise KeyError(sorted(pins - set(result["ports"])))
    return result


def loop(prepared, material, legs, config):
    import numpy as np

    segments = [route(prepared, a, b, config["pitch_mm"], config) for a, b in legs]
    curve = np.vstack([r[0] for r in segments] + [segments[0][0][0]])
    dl = np.concatenate(
        [np.linalg.norm(np.diff(r[0], axis=0), axis=1) for r in segments]
    )
    widths = np.concatenate([r[1] for r in segments]) * config.get("width_scale", 1)
    indices = np.concatenate([r[2] for r in segments])
    foil = [material["copper_mm"][l] for l in material["depth_mm"]]
    area = np.where(
        indices < 0,
        math.pi * widths * material["plating_mm"],
        widths * np.array(foil)[indices],
    )
    rdc = material["rho_ohm_m"] * 1000 * np.sum(dl / area)
    radius = config.get("gmd_width_factor", 0.2235) * dl.sum() / np.sum(dl / widths)
    return {
        "path_mm": curve.tolist(),
        "radius_mm": float(radius),
        "resistance_ohm": float(rdc),
        "inductance_nH": neumann(curve, radius, config["segment_mm"]),
        "widths_mm": widths.tolist(),
    }


def mutual_nh(first, second, step):
    import numpy as np
    from scipy.spatial.distance import cdist

    if (
        first["path_mm"] == second["path_mm"]
        and first["radius_mm"] == second["radius_mm"]
    ):
        return first["inductance_nH"]
    a, b = [resample(np.asarray(obj["path_mm"]), step) for obj in (first, second)]
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    qa, qb = (a[:-1] + a[1:]) / 2, (b[:-1] + b[1:]) / 2
    r = math.sqrt(first["radius_mm"] * second["radius_mm"])
    total = 0.0
    for start in range(0, len(da), 128):
        total += np.sum(
            (da[start : start + 128] @ db.T)
            / np.sqrt(cdist(qa[start : start + 128], qb, "sqeuclidean") + r * r)
        )
    return float(0.1 * total)
