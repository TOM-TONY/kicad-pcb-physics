"""Routed quasi-TEM sections, delays and cascaded frequency responses."""

import math

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq
from shapely.geometry import LineString, Point

from .cross_section import C0, modal, section
from .line_network import response, touchstone
from .line_route import detect, point, reverse, route
from .mesh import polygon


def _cross(widths, gap, geometry, spec):
    return section(
        tuple(widths),
        gap,
        geometry["h"],
        geometry["upper"],
        geometry["t"],
        geometry["er"],
        geometry["tan"],
        geometry["mask"],
        geometry["mask_er"],
        geometry["mask_tan"],
        spec.get("cells", 16),
        spec.get("margin", 16),
    )


def _media(mat, spec, layer):
    cfg = spec["stack"][layer]
    z = mat["depth_mm"]
    plane = cfg["plane"]
    upper = cfg.get("second_plane")
    if upper and (z[upper] - z[layer]) * (z[plane] - z[layer]) >= 0:
        raise ValueError("Stripline requires reference planes on opposite sides")
    return dict(
        h=abs(z[plane] - z[layer]),
        upper=abs(z[upper] - z[layer]) if upper else 0,
        t=mat["copper_mm"][layer],
        er=cfg["er"],
        tan=cfg.get("tan_delta", 0),
        mask=cfg.get("mask_mm", 0),
        mask_er=cfg.get("mask_er", 1),
        mask_tan=cfg.get("mask_tan_delta", 0),
    )


def _geometry(g, mat, spec, name):
    edges = route(g, mat, name, spec.get("endpoints", {}).get(name))
    inside = spec.get("packages", {}).get(name, {})
    if inside:
        if (
            inside["length_mm"] < 0
            or inside["velocity_m_s"] <= 0
            or inside["z_ohm"] <= 0
        ):
            raise ValueError("Invalid package model")
        if inside["length_mm"]:
            e = edges[-1]
            edges.append(
                dict(
                    kind="package",
                    a=e["b"],
                    b=e["b"],
                    layer=e["to_layer"],
                    to_layer=e["to_layer"],
                    width_mm=e["width_mm"],
                    **inside,
                )
            )
    return edges


def _uniform(z, v, tangent):
    c = np.array([[1 / (z * v)]])
    return c, np.array([[z / v]]), c * tangent


def _gain_db(s):
    return -20 * math.log10(max(abs(s), 1e-300))


def _guidance(seg, spec, f, rho, rough):
    from .line_network import rl

    guide = spec.get("design", {})
    if not guide:
        return {}
    pair = len(seg["widths"]) == 2
    geom = seg["medium"]

    def impedance(w, gap, n):
        c, l, _ = _cross([w] * n, gap, geom, spec)
        return modal(c, l)["diff" if n == 2 else "single"]["z_ohm"]

    def match(func, bounds, target):
        lo, hi = bounds
        if lo <= 0 or hi <= lo:
            raise ValueError("Invalid search interval")
        vals = [func(lo), func(hi)]
        if min(vals) <= target <= max(vals):
            return float(brentq(lambda x: func(x) - target, lo, hi, xtol=1e-5))
        return None

    out = {}
    w = seg["widths"][0]
    if "single_ohm" in guide:
        w = match(
            lambda x: impedance(x, 0, 1), guide["width_range_mm"], guide["single_ohm"]
        )
        out["width_mm"] = w
    if w and pair and "diff_ohm" in guide:
        out["fixed_width_mm"] = w
        out["gap_mm"] = match(
            lambda gap: impedance(w, gap, 2), guide["gap_range_mm"], guide["diff_ohm"]
        )
    v = seg["speeds"][0]
    if "skew_ps" in guide:
        out["length_tolerance_mm"] = guide["skew_ps"] * v * 1e-9
    if "loss_db" in guide:
        # Matched uniform line: attenuation follows the differential (odd) mode.
        mode = np.array([1.0, -1.0]) if pair else np.array([1.0])

        def odd(m):
            return (mode @ m @ mode) / (mode @ mode)

        z = odd(np.diag(rl(seg, f, rho, rough)) + 2j * math.pi * f * seg["L"])
        y = odd(2 * math.pi * f * (seg["loss"] + 1j * seg["C"]))
        attenuation = 8.685889638 * np.sqrt(z * y).real
        out["matched_loss_db_per_m"] = float(attenuation)
        out["max_length_mm"] = (
            guide["loss_db"] * 1000 / attenuation if attenuation > 0 else None
        )
        out["frequency_hz"] = f
    return out


def analyze(g, mat, spec, names, folder, index):
    if len(names) not in (1, 2) or len(set(names)) != len(names):
        raise ValueError("Specify a single net or two distinct pair members")
    routes = [_geometry(g, mat, spec, name) for name in names]
    if len(names) == 2 and not spec.get("endpoints", {}).get(names[1]):
        anchor = routes[0][0]["a"]
        if math.dist(anchor, routes[1][-1]["b"]) < math.dist(anchor, routes[1][0]["a"]):
            routes[1] = reverse(routes[1])
    ends = [np.cumsum([0] + [e["length_mm"] for e in edges]) for edges in routes]
    totals = np.array([p[-1] for p in ends])
    step = spec.get("step_mm", 0.25)
    if step <= 0 or min(totals) <= 0:
        raise ValueError("Positive path and step required")
    # Register path positions along the physical propagation axis. Extra length
    # in a detour is then reported at that detour, not spread over the coupon.
    axis = np.asarray(routes[0][-1]["b"]) - routes[0][0]["a"]
    projected = []
    distances = []
    can_project = np.linalg.norm(axis) > 1e-9
    if can_project:
        axis = axis / np.linalg.norm(axis)
    for i, edges in enumerate(routes):
        ss = []
        xy = []
        for e, s0 in zip(edges, ends[i]):
            count = max(1, math.ceil(e["length_mm"] / step)) if "mid" in e else 1
            for u in np.arange(count) / count:
                ss.append(s0 + e["length_mm"] * u)
                xy.append(point(e, u))
        ss.append(totals[i])
        xy.append(edges[-1]["b"])
        ss = np.asarray(ss)
        v = np.asarray(xy) @ axis
        if v[-1] - v[0] <= 1e-9 or np.min(np.diff(v)) < -1e-7:
            can_project = False
        v = (v - v[0]) / max(v[-1] - v[0], 1e-9) + 1e-6 * ss / totals[i]
        projected.append(v / v[-1] if v[-1] > 0 else v)
        distances.append(ss)
    method = "projection" if can_project else "fractional arclength"
    anchors = spec.get("landmarks_mm")
    if anchors:
        if len(names) != 2:
            raise ValueError("Landmarks require a pair")
        m = np.vstack([np.zeros(2), anchors, totals])
        if np.any(np.diff(m, axis=0) <= 0):
            raise ValueError("Increasing interior landmarks required")
        distances = list(m.T)
        projected = [np.linspace(0, 1, len(m))] * 2
        method = "declared corresponding landmarks"
    elif not can_project:
        projected = [np.array([0.0, 1.0]) for _ in names]
        distances = [np.array([0.0, v]) for v in totals]

    def station(i, s):
        return round(float(np.interp(s, distances[i], projected[i])), 13)

    def distance(i, u):
        return float(np.interp(u, projected[i], distances[i]))

    planes = {
        l: polygon(g["nets"][spec["reference"]]["copper"].get(l, []))
        for l in g["layers"]
    }
    # Add exact straight trace/plane-boundary intersections to sampling stations.
    marks = {0.0, 1.0}

    def subdivide(i, e, start):
        count = max(1, math.ceil(e["length_mm"] / step))
        for u in np.linspace(0, 1, count + 1):
            marks.add(station(i, start + e["length_mm"] * u))
        if e["kind"] != "track":
            return
        cfg = spec["stack"][e["layer"]]
        refs = [cfg["plane"]] + (
            [cfg["second_plane"]] if cfg.get("second_plane") else []
        )
        n = count if "mid" in e else 1
        for j in range(n):
            line = LineString([point(e, j / n), point(e, (j + 1) / n)])
            if not line.length:
                continue
            for layer in refs:
                if planes[layer].is_empty:
                    continue
                cut = planes[layer].boundary.intersection(line)
                for part in getattr(cut, "geoms", [cut]):
                    for xy in getattr(part, "coords", []):
                        u = (j + line.project(Point(xy)) / line.length) / n
                        marks.add(station(i, start + u * e["length_mm"]))

    for i, edges in enumerate(routes):
        for e, start in zip(edges, ends[i]):
            subdivide(i, e, start)
    marks = sorted(marks)
    timeline = []
    sections = []
    times = np.zeros(len(names))
    prev_width = [None] * len(names)
    prev_gap = None
    for left, right in zip(marks, marks[1:]):
        if right - left < 1e-12:
            continue
        centre = (left + right) / 2
        media = []
        widths = []
        pos = []
        direction = []
        tags = []
        parts = []
        stubs = []
        layers = []
        for i, edges in enumerate(routes):
            s = distance(i, centre)
            k = min(np.searchsorted(ends[i], s, side="right") - 1, len(edges) - 1)
            e = edges[k]
            u = (s - ends[i][k]) / e["length_mm"]
            xy = point(e, u)
            pos.append(xy)
            layers.append(e["layer"])
            m = _media(mat, spec, e["layer"])
            media.append(m)
            w = e["width_mm"]
            widths.append(w)
            delta = point(e, min(u + 0.001, 1)) - point(e, max(u - 0.001, 0))
            direction.append(delta / max(np.linalg.norm(delta), 1e-12))
            if prev_width[i] is not None and abs(w - prev_width[i]) > 1e-8:
                tags.append(f"{names[i]}:width-change")
            prev_width[i] = w
            if e["kind"] == "track":
                cfg = spec["stack"][e["layer"]]
                for pl in [cfg["plane"]] + (
                    [cfg["second_plane"]] if cfg.get("second_plane") else []
                ):
                    if not planes[pl].covers(Point(xy)):
                        tags.append(f"{names[i]}:reference-gap:{pl}")
                parts.append(_cross([w], 0, m, spec))
            else:
                tags.append(f"{names[i]}:{e['kind']}")
                z = e.get("z_ohm", spec.get("via_z_ohm", 50))
                v = e.get("velocity_m_s", C0 / math.sqrt(m["er"]))
                parts.append(_uniform(z, v, m["tan"]))
            stub = (
                e.get("stubs_mm", [])
                if abs(distance(i, right) - ends[i][k + 1]) < 1e-6
                else []
            )
            stubs.append([v for v in stub if v > 1e-7])
            if stubs[-1]:
                tags.append(f"{names[i]}:via-stub")
        gap = None
        coupled = False
        if len(names) == 2:
            gap = (
                abs(np.cross(np.r_[direction[0], 0], np.r_[pos[1] - pos[0], 0])[2])
                - sum(widths) / 2
            )
            longitudinal = abs((pos[1] - pos[0]) @ direction[0])
            coupled = (
                layers[0] == layers[1]
                and direction[0] @ direction[1] > 0.98
                and gap > 0
                and gap < spec.get("coupling_range_mm", 4 * media[0]["h"])
                and longitudinal < spec.get("registration_mm", step * 2)
                and not any("reference-gap" in t for t in tags)
            )
            if not coupled:
                tags.append("uncoupled")
            if coupled and prev_gap is not None and abs(gap - prev_gap) > 1e-4:
                tags.append("gap-change")
            prev_gap = gap if coupled else None
        if coupled:
            c, l, loss = _cross(widths, round(float(gap), 5), media[0], spec)
        else:
            c, l, loss = [block_diag(*v) for v in zip(*parts)]
        modes = modal(c, l)
        speeds = [modal(p[0], p[1])["single"]["velocity_m_s"] for p in parts]
        if coupled:
            # Odd-mode timing on parallel symmetric sections.
            speeds = [modes["diff"]["velocity_m_s"]] * 2
        end_mm = np.array([distance(i, right) for i in range(len(names))])
        lengths = end_mm - np.array([distance(i, left) for i in range(len(names))])
        times += lengths / np.array(speeds) * 1e9
        known = not any("reference-gap" in tag for tag in tags)
        timeline.append(
            dict(
                reference_resolved=known,
                local_z_ohm=(
                    {k: v["z_ohm"] for k, v in modes.items()} if known else None
                ),
                end_mm=end_mm.tolist(),
                increment_mm=lengths.tolist(),
                xy=[v.tolist() for v in pos],
                layers=layers,
                width_mm=widths,
                gap_mm=float(gap) if gap is not None and coupled else None,
                end_xy=[
                    point(
                        edges[k],
                        np.clip(
                            (distance(i, right) - ends[i][k]) / edges[k]["length_mm"],
                            0,
                            1,
                        ),
                    ).tolist()
                    for i, edges in enumerate(routes)
                    for k in [
                        min(
                            np.searchsorted(ends[i], distance(i, centre), side="right")
                            - 1,
                            len(edges) - 1,
                        )
                    ]
                ],
                coupled=bool(coupled),
                single_ohm=[modal(p[0], p[1])["single"]["z_ohm"] for p in parts],
                modes=modes,
                time_ps=times.tolist(),
                skew_ps=float(times[0] - times[-1]),
                flags=tags,
                stubs_mm=stubs,
            )
        )
        sections.append(
            dict(
                C=c,
                L=l,
                loss=loss,
                widths=widths,
                thickness=[m["t"] for m in media],
                heights=[m["h"] for m in media],
                medium=media[0],
                lengths=lengths.tolist(),
                speeds=speeds,
                stub_z=[spec.get("via_z_ohm", 50)] * len(names),
                stubs=stubs,
            )
        )
    f = spec["frequency_hz"]
    zref = spec.get("reference_ohm", 50)
    if not f or min(f) <= 0 or list(f) != sorted(set(f)) or zref <= 0:
        raise ValueError("Positive increasing frequencies and reference required")
    rho = mat.get("rho_ohm_m", 1.7241e-8)
    rough = spec.get("roughness_um", 0)
    if rough < 0:
        raise ValueError("Negative surface roughness")
    se = []
    mixed = []
    sweep = []
    for frequency in f:
        s, sm, loss = response(sections, frequency, zref, rho, rough)
        se.append(s)
        if sm is not None:
            mixed.append(sm)
        forward = s[1, 0] if sm is None else sm[2, 0]

        def isolation(copper, dielectric):
            sk, mk, _ = response(
                sections, frequency, zref, rho, rough, copper, dielectric
            )
            return _gain_db(sk[1, 0] if mk is None else mk[2, 0])

        lossless = isolation(False, False)
        sweep.append(
            dict(
                frequency_hz=frequency,
                insertion_loss_db=_gain_db(forward),
                lossless_insertion_db=lossless,
                conductor_increment_db=isolation(True, False) - lossless,
                dielectric_increment_db=isolation(False, True) - lossless,
                conductor_dielectric_estimate_db=loss.tolist(),
            )
        )
    prefix = f"line-{index:02d}"
    files = [f"{prefix}.s{len(names) * 2}p"]
    touchstone(folder / files[0], f, se, zref)
    if mixed:
        files.append(f"{prefix}-mixed.ts")
        touchstone(folder / files[-1], f, mixed, zref, True)
    label = " / ".join(names)
    maps = []
    for layer in g["layers"]:
        for idx, net in enumerate(names):
            rows = [r for r in timeline if r["layers"][idx] == layer]
            for key, unit, points in [
                (
                    "single impedance",
                    "ohm",
                    [
                        [*r["xy"][idx], r["single_ohm"][idx]]
                        for r in rows
                        if r["reference_resolved"]
                    ],
                ),
                ("delay", "ps", [[*r["end_xy"][idx], r["time_ps"][idx]] for r in rows]),
                ("width", "mm", [[*r["xy"][idx], r["width_mm"][idx]] for r in rows]),
            ]:
                if points:
                    maps.append(
                        dict(
                            name=net + " / " + key,
                            unit=unit,
                            layer=layer,
                            points=points,
                        )
                    )
        if len(names) == 2:
            rows = [r for r in timeline if r["layers"][0] == layer]
            for mode in ("diff", "common"):
                pts = [
                    [*r["xy"][0], r["modes"][mode]["z_ohm"]]
                    for r in rows
                    if r["reference_resolved"] and r["coupled"]
                ]
                if pts:
                    maps.append(
                        dict(
                            name=label + " / " + mode,
                            unit="ohm",
                            layer=layer,
                            points=pts,
                        )
                    )
            if rows:
                maps.append(
                    dict(
                        name=label + " / skew",
                        unit="ps",
                        layer=layer,
                        points=[[*r["end_xy"][0], r["skew_ps"]] for r in rows],
                    )
                )
        broken = [
            [*r["xy"][i], 1]
            for r in timeline
            for i, name in enumerate(names)
            if r["layers"][i] == layer
            and any(flag.startswith(name + ":reference-gap:") for flag in r["flags"])
        ]
        if broken:
            maps.append(
                dict(
                    name=label + " / Missing reference",
                    unit="flag",
                    layer=layer,
                    points=broken,
                )
            )
    complete = all(r["reference_resolved"] for r in timeline)
    guide_seg = next(
        (s for s, r in zip(sections, timeline) if r["coupled"] and not r["flags"]),
        sections[0],
    )
    return dict(
        name=label,
        nets=names,
        start_xy=[edges[0]["a"] for edges in routes],
        registration=method,
        length_mm=totals.tolist(),
        time_ps=times.tolist(),
        skew_ps=float(times[0] - times[-1]),
        skew_length_mm=float(totals[0] - totals[-1]),
        route_lengths_by_kind_mm=[
            {
                kind: sum(e["length_mm"] for e in path if e["kind"] == kind)
                for kind in ("track", "via", "package")
            }
            for path in routes
        ],
        sections=timeline,
        sweep=sweep,
        files=files,
        propagation_model="Local quasi-TEM cascade"
        if complete
        else "Experimental continuation of ideal reference through flagged gaps; actual gap impedance and loss unresolved",
        coupled_asymmetry="Projected odd/common impedances; mode conversion in cascade uses full matrices",
        design=_guidance(guide_seg, spec, f[-1], rho, rough),
        maps=maps,
    )


def studies(g, mat, spec, folder):
    groups = spec.get("routes", [])
    if spec.get("detect_pairs"):
        groups = groups + [
            p for p in detect(g["nets"], spec.get("pair_suffixes")) if p not in groups
        ]
    results = [
        analyze(g, mat, spec, ([v] if isinstance(v, str) else v), folder, i)
        for i, v in enumerate(groups)
    ]
    by_net = {name: (r, i) for r in results for i, name in enumerate(r["nets"])}
    matches = []
    for first, second in spec.get("match", []):
        a, i = by_net[first]
        b, j = by_net[second]
        xy = []
        lengths = []
        delay = []
        for result, idx in ((a, i), (b, j)):
            xy.append(
                np.array(
                    [result["start_xy"][idx]]
                    + [r["end_xy"][idx] for r in result["sections"]]
                )
            )
            lengths.append(
                np.array([0] + [r["end_mm"][idx] for r in result["sections"]])
            )
            delay.append(
                np.array([0] + [r["time_ps"][idx] for r in result["sections"]])
            )
        axis = xy[0][-1] - xy[0][0]
        proj = [(v - v[0]) @ axis for v in xy]
        spatial = all(row[-1] > 0 and min(np.diff(row)) >= -1e-7 for row in proj)
        pk = (
            [(v / v[-1] + 1e-6 * s / s[-1]) / (1 + 1e-6) for v, s in zip(proj, lengths)]
            if spatial
            else [s / s[-1] for s in lengths]
        )
        coords = np.unique(np.concatenate(pk))
        dlen = np.interp(coords, pk[0], lengths[0]) - np.interp(
            coords, pk[1], lengths[1]
        )
        dt = np.interp(coords, pk[0], delay[0]) - np.interp(coords, pk[1], delay[1])
        profile = [
            {"station": u, "length_difference_mm": x, "delay_difference_ps": y}
            for u, x, y in zip(coords, dlen, dt)
        ]
        matches.append(
            {
                "nets": [first, second],
                "registration": "projection" if spatial else "fractional arclength",
                "stations": profile,
                "length_difference_mm": a["length_mm"][i] - b["length_mm"][j],
                "delay_difference_ps": a["time_ps"][i] - b["time_ps"][j],
            }
        )
    return results, matches
