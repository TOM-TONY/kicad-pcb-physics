"""Steady layered-board FEM with a common, interface-conforming triangular mesh.

The in-plane copper pattern, part contacts and barrel walls share vertices on
all foil levels. Dielectric thickness is lumped onto neighbouring foils in
plane. Through-thickness transfer, ambient films and body contacts use lumped
linear-triangle mass matrices. Board edges are insulated unless clamped.
"""

import numpy as np
import pyamg
import shapely as sh
from scipy import sparse
from scipy.sparse.linalg import cg

from .heat import _positive, _shape


def _lattice(bounds, step):
    x0, y0, x1, y1 = bounds
    xs = np.arange(np.ceil(x0 / step), np.floor(x1 / step) + 1) * step
    ys = np.arange(np.ceil(y0 / step), np.floor(y1 / step) + 1) * step
    return [sh.LineString([(x, y0), (x, y1)]) for x in xs] + [
        sh.LineString([(x0, y), (x1, y)]) for y in ys
    ]


def _geometry(geo, settings, snap, tolerance):
    def clean(poly):
        return sh.set_precision(poly.simplify(tolerance, preserve_topology=True), snap)

    layers = geo["layers"]
    nets = geo["nets"].values()
    raw_board = _shape(geo["outline"])
    board = clean(raw_board)
    if board.area <= 0 or not board.is_valid:
        raise ValueError("Invalid board after geometry snapping")
    metal = []
    area_change = {}
    for layer in layers:
        raw = sh.union_all(
            [_shape(n["copper"].get(layer, [])) for n in nets]
        ).intersection(board)
        metal.append(clean(raw))
        area_change[layer] = metal[-1].area - raw.area
    contacts, labels = {}, {}
    for ref in settings.get("part_power_w", {}):
        part = geo["parts"][ref]
        layer = part["layer"]
        explicit = settings.get("part_pad_ids", {}).get(ref)
        selected = {
            pin: [p for p in pads if p["layer"] == layer]
            for n in nets
            for pin, pads in n.get("pads", {}).items()
            if (pin in explicit if explicit is not None else pin.startswith(ref + "."))
        }
        if explicit is not None and any(not selected.get(pin) for pin in explicit):
            raise ValueError(f"Missing mounting-layer contact for {ref}")
        polygons = [
            polygon
            for pads in selected.values()
            for pad in pads
            for polygon in pad["polygons"]
        ]
        if polygons:
            contacts[ref] = clean(
                _shape(polygons)
                .intersection(board)
                .intersection(metal[layers.index(layer)])
            )
            labels[ref] = "pad union"
        else:
            contacts[ref] = clean(sh.box(*part["bounds"]).intersection(board))
            labels[ref] = "footprint bounds"
        if contacts[ref].area <= 0:
            raise ValueError(f"Empty body contact for {ref}")

    plating = _positive(settings.get("via_plating_um", 20), "via plating", True) * 0.001
    by_span = {}
    for n in nets:
        for via in n.get("vias", []):
            if not plating:
                continue
            indices = sorted({layers.index(name) for name in via["layers"]})
            if len(indices) < 2:
                raise ValueError("A barrel needs two or more physical layer levels")
            radius = _positive(via["drill_mm"], "drill") / 2
            bore = sh.Point(via["xy"]).buffer(radius, quad_segs=16)
            wall = bore.buffer(plating, quad_segs=4).difference(bore)
            by_span.setdefault((indices[0], indices[-1]), []).append(wall)
    walls = {
        span: clean(sh.union_all(values).intersection(board))
        for span, values in by_span.items()
    }
    return board, metal, contacts, walls, labels, area_change


def _triangulate(geo, board, shapes, refs, grid, near, margin, snap):
    lines = [poly.boundary for poly in (board, *shapes) if not poly.is_empty]
    lines.extend(_lattice(board.bounds, grid))
    for ref in refs:
        bounds = np.array(geo["parts"][ref]["bounds"]) + [
            -margin,
            -margin,
            margin,
            margin,
        ]
        bounds[:2] = np.floor(bounds[:2] / grid) * grid
        bounds[2:] = np.ceil(bounds[2:] / grid) * grid
        lines.extend(_lattice(bounds, near))
    network = sh.union_all(lines, grid_size=snap)
    faces = sh.get_parts(sh.polygonize(sh.get_parts(network)))
    sh.prepare(board)
    faces = faces[sh.covers(board, sh.point_on_surface(faces))]
    elements = sh.get_parts(sh.constrained_delaunay_triangles(faces))
    coords = sh.get_coordinates(elements).reshape(-1, 4, 2)[:, :3]
    quantized = np.rint(coords / snap).astype(np.int64)
    vertices, ids = np.unique(quantized.reshape(-1, 2), axis=0, return_inverse=True)
    points = vertices * snap
    cells = ids.reshape(-1, 3).astype(np.int32)
    tri = points[cells]
    e, f = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]) / 2
    if np.any(area <= 0) or abs(area.sum() - board.area) > 1e-6 * board.area:
        raise ArithmeticError("Triangle coverage does not match the board area")
    edges = np.sort(
        np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1
    )
    unique, count = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[count == 1]
    length = np.linalg.norm(
        points[boundary[:, 0]] - points[boundary[:, 1]], axis=1
    ).sum()
    if count.max() > 2 or abs(length - board.length) > 1e-6 * max(1, board.length):
        raise ArithmeticError("Unmatched internal triangle edges")
    return points, cells, tri, area, elements


def _dc_load(power, points, cells, elements, results, layers, width):
    # Only simultaneous DC cases belong here. Group coincident source squares
    # across layers, then integrate each linear basis at its clipped centroid.
    sources = {}
    for result in results:
        for src in result["heat_cells"]:
            key = (src["x_mm"], src["y_mm"])
            watts = sources.setdefault(key, np.zeros(len(layers)))
            watts[layers.index(src["layer"])] += _positive(
                src["watts"], "DC power", True
            )
    if not sources:
        return 0.0
    tree = sh.STRtree(elements)
    for (x, y), watts in sources.items():
        if not watts.any():
            continue
        h = width / 2
        patch = sh.box(x - h, y - h, x + h, y + h)
        ids = tree.query(patch, predicate="intersects")
        clipped = sh.intersection(elements[ids], patch)
        areas = sh.area(clipped)
        valid = areas > 0
        if not valid.any():
            raise ValueError("DC source has no positive area on the board")
        area = areas[valid]
        indices = cells[ids[valid]]
        tri = points[indices]
        xy = sh.get_coordinates(sh.centroid(clipped[valid]))
        e, f, v = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], xy - tri[:, 0]
        det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
        u = (v[:, 0] * f[:, 1] - v[:, 1] * f[:, 0]) / det
        w = (e[:, 0] * v[:, 1] - e[:, 1] * v[:, 0]) / det
        weights = (area / area.sum())[:, None] * np.column_stack((1 - u - w, u, w))
        for z in np.flatnonzero(watts):
            np.add.at(power[z], indices.ravel(), (watts[z] * weights).ravel())
    return float(sum(v.sum() for v in sources.values()))


def analyze(geo, settings, dc_results):
    """Solve the supplied geometry and return full nodal temperature maps.

    grid_mm sets the background partition; near_parts_mm sets its spacing by
    powered footprints (near_part_margin_mm). snap_mm sets the coordinate
    precision, and simplify_mm controls a fixed boundary simplification.
    Inspect reported area changes when the smallest feature approaches either
    tolerance. Re-run with different partition sizes for a refinement check.
    Other source, contact, stack and boundary settings follow heat.analyze.
    """
    layers = list(geo["layers"])
    nz = len(layers)
    if nz < 2 or len(set(layers)) != nz:
        raise ValueError("At least two unique copper layers are required")
    grid = _positive(settings.get("grid_mm", 1), "grid_mm")
    near = _positive(settings.get("near_parts_mm", grid / 4), "near_parts_mm")
    snap = _positive(settings.get("snap_mm", 0.001), "snap_mm")
    tolerance = _positive(settings.get("simplify_mm", 0.001), "simplify_mm", True)
    margin = _positive(
        settings.get("near_part_margin_mm", 1), "near_part_margin_mm", True
    )
    if near < snap or near > grid:
        raise ValueError("Require snap_mm <= near_parts_mm <= grid_mm")
    copper = settings.get("copper_um", [35] * nz)
    if isinstance(copper, dict):
        copper = [copper[name] for name in layers]
    cu = np.array([_positive(t, "copper thickness") for t in copper]) * 1e-6
    gaps = (
        np.array(
            [
                _positive(t, "dielectric thickness")
                for t in settings.get("dielectric_mm", [1.5 / (nz - 1)] * (nz - 1))
            ]
        )
        * 1e-3
    )
    if len(cu) != nz or len(gaps) != nz - 1:
        raise ValueError("Invalid stack dimensions")
    kd = _positive(settings.get("dielectric_w_mk", 0.3), "dielectric conductivity")
    kcu = _positive(settings.get("copper_w_mk", 390), "copper conductivity")
    depth = (np.r_[0, gaps] + np.r_[gaps, 0]) / 2
    board, metal, pads, walls, labels, simplification = _geometry(
        geo, settings, snap, tolerance
    )
    points, cells, tri, area, elements = _triangulate(
        geo,
        board,
        [*metal, *pads.values(), *walls.values()],
        pads,
        grid,
        near,
        margin,
        snap,
    )
    n = len(points)
    centroids = tri.mean(axis=1)

    def lump(weights):
        return np.bincount(
            cells.ravel(), weights=np.repeat(weights / 3, 3), minlength=n
        )

    def flags(poly):
        sh.prepare(poly)
        return sh.contains_xy(poly, centroids[:, 0], centroids[:, 1])

    nodal_area = lump(area) * 1e-6
    wall_area = np.zeros((nz - 1, n))
    wall_error = {}
    for (lo, hi), wall in walls.items():
        contribution = lump(area * flags(wall)) * 1e-6
        wall_area[lo:hi] += contribution
        wall_error[f"{layers[lo]} to {layers[hi]}"] = float(
            contribution.sum() * 1e6 - wall.area
        )
    if np.any(wall_area > nodal_area * (1 + 1e-8)):
        raise ValueError("Overlapping barrel sections exceed available area")

    # Dot products of rotated edge vectors equal dot products of shape gradients
    # times (2*area)^2. The in-plane mm factors cancel for a sheet in W/K.
    vectors = np.roll(tri, -1, axis=1) - np.roll(tri, -2, axis=1)
    gram = np.einsum("tik,tjk->tij", vectors, vectors) / (4 * area[:, None, None])
    ii, jj = np.repeat(cells, 3, axis=1).ravel(), np.tile(cells, (1, 3)).ravel()
    vertical = [
        (nodal_area - wall_area[z]) * kd / gap
        + wall_area[z] * kcu / (gap + (cu[z] + cu[z + 1]) / 2)
        for z, gap in enumerate(gaps)
    ]
    blocks = [[None for _ in layers] for _ in layers]
    copper_error = {}
    for z, layer in enumerate(layers):
        covered = flags(metal[z])
        copper_error[layer] = float(area[covered].sum() - metal[z].area)
        sheet = kd * depth[z] + kcu * cu[z] * covered
        k = sparse.coo_matrix(
            ((gram * sheet[:, None, None]).ravel(), (ii, jj)), shape=(n, n)
        ).tocsr()
        diag = np.zeros(n)
        if z:
            diag += vertical[z - 1]
        if z < nz - 1:
            diag += vertical[z]
            blocks[z][z + 1] = blocks[z + 1][z] = -sparse.diags(
                vertical[z], format="csr"
            )
        blocks[z][z] = k + sparse.diags(diag, format="csr")
    plate = sparse.bmat(blocks, format="csr")
    del blocks, k, gram, vectors, ii, jj, tri, vertical, wall_area
    powers = settings.get("part_power_w", {})
    parts = list(powers)
    power = np.zeros((nz, n))
    results = dc_results.values() if isinstance(dc_results, dict) else dc_results
    dc_w = _dc_load(
        power,
        points,
        cells,
        elements,
        results,
        layers,
        _positive(settings.get("dc_cell_mm", 1), "DC source width"),
    )
    if not np.isclose(power.sum(), dc_w, rtol=1e-9, atol=1e-12):
        raise ArithmeticError("DC heat source integration failed")
    for layer, watts in settings.get("layer_power_w", {}).items():
        power[layers.index(layer)] += (
            nodal_area / nodal_area.sum() * _positive(watts, "layer power", True)
        )
    rhs = np.r_[
        power.ravel(), [_positive(powers[r], "body power", True) for r in parts]
    ]
    size = len(rhs)
    sinks = np.zeros(size)
    sinks[:n] = nodal_area * _positive(
        settings.get("h_top_w_m2k", 6), "top convection", True
    )
    sinks[(nz - 1) * n : nz * n] = nodal_area * _positive(
        settings.get("h_bottom_w_m2k", 6), "bottom convection", True
    )
    matrix = sparse.block_diag(
        (plate, sparse.csr_matrix((len(parts), len(parts)))), format="csr"
    )
    del plate, elements, power
    a, b, v = [], [], []
    contact_error = {}
    for j, ref in enumerate(parts, nz * n):
        weights = lump(area * flags(pads[ref]))
        contact_error[ref] = float(weights.sum() - pads[ref].area)
        ix = np.flatnonzero(weights)
        if not len(ix):
            raise ArithmeticError(f"No meshed mounting contact for {ref}")
        rc = settings.get("contact_k_w", 12)
        r = _positive(
            rc[ref] if isinstance(rc, dict) else rc, "body contact resistance"
        )
        links = weights[ix] / (weights.sum() * r)
        nodes = ix + n * layers.index(geo["parts"][ref]["layer"])
        body = np.full(len(ix), j)
        a.extend([nodes, nodes, body, [j]])
        b.extend([nodes, body, nodes, [j]])
        v.extend([links, -links, -links, [links.sum()]])
        if ref in settings.get("part_to_air_k_w", {}):
            sinks[j] = 1 / _positive(
                settings["part_to_air_k_w"][ref], "body to air resistance"
            )
    if parts:
        matrix += sparse.coo_matrix(
            (np.concatenate(v), (np.concatenate(a), np.concatenate(b))),
            shape=(size, size),
        ).tocsr()
    matrix += sparse.diags(sinks, format="csr")
    fixed_edges = set(settings.get("fixed_edges", []))
    boundary = dict(
        left=(0, board.bounds[0]),
        right=(0, board.bounds[2]),
        top=(1, board.bounds[1]),
        bottom=(1, board.bounds[3]),
    )
    if fixed_edges - boundary.keys() or (
        fixed_edges and not board.equals(sh.box(*board.bounds))
    ):
        raise ValueError("Fixed edges must name faces of a rectangular outline")
    pinned = np.zeros(n, dtype=bool)
    for edge in fixed_edges:
        axis, coordinate = boundary[edge]
        pinned |= np.abs(points[:, axis] - coordinate) < snap / 4
    pinned = np.r_[np.tile(pinned, nz), np.zeros(len(parts), dtype=bool)]
    count, islands = sparse.csgraph.connected_components(matrix, directed=False)
    if np.any(np.bincount(islands, weights=sinks + pinned, minlength=count) == 0):
        raise ValueError("A thermal island has no heat sink")
    free = ~pinned
    k = matrix[free][:, free] if pinned.any() else matrix
    rise = np.zeros(size)
    if free.any() and np.any(rhs[free]):
        scale = 1 / np.sqrt(k.diagonal())
        balanced = k.multiply(scale[:, None]).multiply(scale[None, :]).tocsr()
        pre = pyamg.ruge_stuben_solver(balanced).aspreconditioner()
        u, status = cg(
            balanced, rhs[free] * scale, M=pre, rtol=1e-9, atol=0, maxiter=900
        )
        if status:
            raise ArithmeticError(f"Conforming heat solver did not converge ({status})")
        rise[free] = u * scale
    cut_w = float((rhs - matrix @ rise)[pinned].sum()) if pinned.any() else 0.0
    convected = float(sinks @ rise)
    imbalance = abs(float(rhs.sum()) - cut_w - convected)
    if not np.isfinite(rise).all() or imbalance > 1e-7 * max(1e-7, rhs.sum()):
        raise ArithmeticError("Heat balance failed")
    ambient = float(settings.get("ambient_c", 25))
    if not np.isfinite(ambient):
        raise ValueError("Non-finite ambient")
    temp = rise + ambient
    iz, ip = divmod(int(temp[: nz * n].argmax()), n)
    maps = [
        dict(
            name="Conforming board temperature",
            layer=layer,
            unit="degC",
            points=np.column_stack((points, temp[z * n : (z + 1) * n])).tolist(),
        )
        for z, layer in enumerate(layers)
    ]
    return dict(
        method="conformal",
        maps=maps,
        peak_board_c=float(temp[iz * n + ip]),
        peak_layer=layers[iz],
        peak_xy_mm=points[ip].tolist(),
        body_c=dict(zip(parts, temp[nz * n :].tolist())),
        contact_model=labels,
        power_w=float(rhs.sum()),
        copper_w=dc_w,
        convected_w=convected,
        cut_w=cut_w,
        balance_error_w=imbalance,
        grid_mm=grid,
        near_parts_mm=near,
        snap_mm=snap,
        simplify_mm=tolerance,
        vertices_per_layer=n,
        triangles=len(area),
        area_mm2=float(area.sum()),
        simplification_area_change_mm2=simplification,
        copper_integration_error_mm2=copper_error,
        contact_integration_error_mm2=contact_error,
        barrel_integration_error_mm2=wall_error,
    )
