"""Conservative through-via placement windows and morphological plane diagnostics."""

import math
import numpy as np
import shapely as sh
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .mesh import polygon
from . import stitch_solver


def _pieces(shape):
    if shape.geom_type == "Polygon":
        return [shape] if not shape.is_empty else []
    return [
        p
        for p in getattr(shape, "geoms", [])
        if p.geom_type == "Polygon" and p.area > 0
    ]


def survey(g, spec):
    required = {"holes", "pad_obstacles", "via_keepouts", "extra_metal"}
    if not required <= g.keys():
        raise ValueError("Re-export the board with drill, pad and keepout obstacles")
    net = spec["net"]
    data = g["nets"][net]
    layers = spec.get("layers", g["layers"])
    if len(layers) < 2 or set(layers) - set(g["layers"]):
        raise ValueError("Invalid stitching layer set")
    radius = spec["land_mm"] / 2
    bore = spec["drill_mm"] / 2
    clear = spec["clearance_mm"]
    edge = spec["edge_clearance_mm"]
    hc = spec["hole_clearance_mm"]
    pitch = spec["search_step_mm"]
    spacing = spec["via_spacing_mm"]
    if min(radius - bore, bore, pitch) <= 0 or min(clear, edge, hc, spacing) < 0:
        raise ValueError("Invalid via dimensions or clearance")
    boundary = polygon(g["outline"])
    window = box(*spec.get("bounds_mm", boundary.bounds))
    # Buffer distances before intersection with the requested window. A keepout
    # just outside the window can still preclude a candidate inside it.
    roi = window.buffer(max(radius + clear, radius + edge, bore + hc, spacing) + pitch)

    def nearby(records):
        return polygon(
            [p for p in records if _bounds_hit(p["shell"], roi.bounds)]
        ).intersection(roi)

    foil = {l: nearby(data["copper"].get(l, [])) for l in g["layers"]}
    allowed = boundary.buffer(-(radius + edge)).intersection(window)
    # The radius cushion bounds the chord error of circular polygon buffers.
    tol = max(0.003, radius * 0.003)
    for l in layers:
        allowed = allowed.intersection(foil[l].buffer(-radius - tol))
    blockers = [foil[layer] for layer in g["layers"] if layer not in layers]
    # A through barrel must not silently pick up omitted ground layers.
    for n, con in g["nets"].items():
        if n != net:
            for l, records in con["copper"].items():
                blockers.append(nearby(records))
    for records in g["extra_metal"].values():
        blockers.append(nearby(records))
    pads = [p for item in g["pad_obstacles"] for p in item["polygons"]]
    blockers.append(nearby(pads))
    # Through-hole lands and barrel can collide anywhere in the full stack.
    allowed = allowed.difference(unary_union(blockers).buffer(radius + clear + tol))
    allowed = allowed.difference(nearby(g["holes"]).buffer(bore + hc + tol))
    for rule in g["via_keepouts"]:
        if set(rule["layers"]) & set(g["layers"]):
            allowed = allowed.difference(nearby(rule["polygons"]).buffer(radius + tol))
    for area in spec.get("exclude_mm", []):
        allowed = allowed.difference(box(*area).buffer(radius + tol))
    vias = [v for con in g["nets"].values() for v in con["vias"]]
    if not allowed.is_empty:
        x0, y0, x1, y1 = allowed.bounds
        xs = np.arange(math.ceil(x0 / pitch), math.floor(x1 / pitch) + 1) * pitch
        ys = np.arange(math.ceil(y0 / pitch), math.floor(y1 / pitch) + 1) * pitch
        xx, yy = np.meshgrid(xs, ys)
        valid = sh.contains_xy(allowed, xx, yy)
        coords = np.column_stack((xx[valid], yy[valid]))
        if vias and len(coords):
            nearest = cKDTree([v["xy"] for v in vias]).query(coords)[0]
            coords = coords[nearest >= spacing]
        coords = np.round(coords, 6).tolist()
    else:
        coords = []
    return foil, coords


def _bounds_hit(points, b):
    if not points:
        return False
    a = np.asarray(points)
    lo = a.min(axis=0)
    hi = a.max(axis=0)
    return lo[0] <= b[2] and hi[0] >= b[0] and lo[1] <= b[3] and hi[1] >= b[1]


def topology(g, spec, with_groups=False):
    """Report connected patches and neck-separated interiors on unclipped copper."""
    net = g["nets"][spec["net"]]
    neck = spec.get("neck_mm", 0.5)
    if neck <= 0:
        raise ValueError("Positive neck width required")
    holes = polygon(g["holes"])
    known = net["vias"]
    patch_shapes = []
    patch_contacts = []
    features = []
    catchments = []
    for layer, records in net["copper"].items():
        copper = polygon(records)
        # Restore just the bores of same-net vias for the topological survey.
        # Leave slots and unrelated drills in place.
        vias = [v for v in known if layer in v["layers"]]
        disks = [Point(v["xy"]).buffer(v["drill_mm"] / 2 + 0.003) for v in vias]
        topology_shape = copper.union(holes.intersection(unary_union(disks)))
        for component in _pieces(topology_shape):
            patch = len(patch_shapes)
            patch_shapes.append(component)
            patch_contacts.append(
                [tuple(v["xy"]) for v in vias if component.covers(Point(v["xy"]))]
            )
            islands = _pieces(component.buffer(-neck / 2))
            regions = [("patch", component)]
            if len(islands) > 1:
                regions.extend(
                    ("neck-lobe", i.buffer(neck / 2).intersection(component))
                    for i in islands
                )
            for kind, shape in regions:
                contacts = [v for v in vias if shape.covers(Point(v["xy"]))]
                p = shape.representative_point()
                dist = min((p.distance(Point(v["xy"])) for v in vias), default=None)
                features.append(
                    dict(
                        kind=kind,
                        patch=patch,
                        nearest_ground_via_mm=min(
                            (
                                p.distance(Point(v["xy"]))
                                for v in vias
                                if v.get("kind", "via") == "via"
                            ),
                            default=None,
                        ),
                        layer=layer,
                        xy=[p.x, p.y],
                        area_mm2=shape.area,
                        ground_contacts=len(contacts),
                        nearest_ground_contact_mm=dist,
                    )
                )
                catchments.append(shape)
    # Use the largest continuous patch as the reference. Vias join all contacted patches.
    roots = list(range(len(patch_shapes)))

    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    first = {}
    for patch, coords in enumerate(patch_contacts):
        for xy in coords:
            roots[root(patch)] = root(first.setdefault(xy, patch))
    groups = [root(i) for i in range(len(roots))]
    anchor = (
        max(range(len(roots)), key=lambda i: patch_shapes[i].area) if roots else None
    )
    for item in features:
        item["connected_to_largest_patch"] = groups[item["patch"]] == groups[anchor]
        item["electrical_group"] = groups[item["patch"]]
    retained = [
        i
        for i, item in enumerate(features)
        if item["kind"] != "patch"
        or item["ground_contacts"] < 2
        or not item["connected_to_largest_patch"]
    ]
    features = [features[i] for i in retained]
    catchments = [catchments[i] for i in retained]
    if with_groups:
        return (
            features,
            catchments,
            patch_shapes,
            groups,
            (groups[anchor] if roots else None),
        )
    return features, catchments


def analyze(g, mat, spec):
    _, candidates = survey(g, spec)
    features, shapes, patches, groups, anchor = topology(g, spec, True)
    controls = {
        **spec,
        "min_spacing_mm": max(
            spec.get("min_spacing_mm", 0),
            spec["via_spacing_mm"],
            spec["drill_mm"] + spec["hole_clearance_mm"],
        ),
    }
    if "signal_path_mm" in spec:
        controls["exciting_path_mm"] = spec["signal_path_mm"]
    if spec["objective"] == "islands":
        contacts = np.array([f["ground_contacts"] for f in features])
        hits = {
            tuple(site): np.array([s.covers(Point(site)) for s in shapes], int)
            for site in candidates
        }
        joins = {
            tuple(site): {
                group
                for patch, group in zip(patches, groups)
                if patch.covers(Point(site))
            }
            for site in candidates
        }

        def deficit(chosen):
            edges = [joins[tuple(p)] for p in chosen]
            linked = {anchor}
            for _ in range(len(edges)):
                for linked_groups in edges:
                    if linked & linked_groups:
                        linked.update(linked_groups)
            count = contacts + sum(
                (hits[tuple(p)] for p in chosen), start=np.zeros(len(shapes), int)
            )
            return int(
                sum(
                    max(2 - int(n), 0) + (r["electrical_group"] not in linked)
                    for r, n in zip(features, count)
                )
            )

        controls["island_score"] = deficit
        controls["island_unit"] = "contact and connectivity deficits"
    result = stitch_solver.analyze(g, mat, controls, candidates)
    known = g["nets"][spec["net"]]["vias"]
    signal = g["nets"].get(spec.get("signal_net"), {})

    def nearest(v):
        xy = v["xy"]
        d = [
            math.dist(xy, u["xy"])
            for u in known
            if u.get("kind", "via") == "via"
            and len(set(u["layers"]) & set(v["layers"])) >= 2
        ]
        return min(d, default=None)

    result["signal_transitions"] = [
        {"xy": v["xy"], "layers": v["layers"], "nearest_return_mm": nearest(v)}
        for v in signal.get("vias", [])
        if v.get("kind", "via") == "via"
    ]
    result["features"] = features
    f = spec.get("max_frequency_hz", spec.get("frequency_hz", 0))
    wavelength = 299792458e3 / (f * math.sqrt(spec["er"])) if f else None
    result["wavelength_mm"] = wavelength
    unique = sorted(set(tuple(v["xy"]) for v in known if v.get("kind", "via") == "via"))
    dist = cKDTree(unique).query(unique, k=2)[0][:, 1] if len(unique) > 1 else []
    result["via_spacing"] = [
        {
            "xy": xy,
            "nearest_mm": float(d),
            "wavelength_fraction": float(d / wavelength) if wavelength else None,
        }
        for xy, d in zip(unique, dist)
    ]
    result["legal_candidates"] = len(candidates)
    for layer in spec.get("layers", g["layers"]):
        for label, unit, pts in [
            (
                "Recommended sequence",
                "step",
                [[*r["xy"], r["step"]] for r in result["rows"]],
            ),
            (
                "Ground patch contacts",
                "count",
                [
                    [*r["xy"], r["ground_contacts"]]
                    for r in features
                    if r["layer"] == layer
                ],
            ),
            (
                "Nearest return",
                "mm",
                [
                    [*r["xy"], r["nearest_return_mm"]]
                    for r in result["signal_transitions"]
                    if layer in r["layers"] and r["nearest_return_mm"] is not None
                ],
            ),
        ]:
            if pts:
                result["maps"].append(
                    dict(
                        name=spec["name"] + " / " + label,
                        unit=unit,
                        layer=layer,
                        points=pts,
                    )
                )
    return result
