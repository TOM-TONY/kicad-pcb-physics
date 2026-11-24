"""Steady, linear layered-board heat flow. Public geometry coordinates are in mm.

Each material tile has a fixed area-averaged in-plane sheet conductance.
Changing the solution grid does not change the material averaging length.
Copper islands inside a tile can therefore spread heat more readily than etched
copper. Maps describe this effective board, not semiconductor junctions.
"""

import math

import numpy as np
import shapely as sh
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


def _shape(records):
    return sh.union_all([sh.Polygon(r["shell"], r.get("holes", [])) for r in records])


def _positive(value, name, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _edges(start, stop, pitch):
    count = int(np.ceil(round((stop - start) / pitch, 10)))
    return np.r_[start + pitch * np.arange(count), stop]


def _areas(shape, cells):
    """Intersect only boundary cells; full interior cells need no clipping."""
    sh.prepare(shape)
    values = np.zeros(cells.shape)
    covered = sh.covers(shape, cells)
    values[covered] = sh.area(cells[covered])
    cut = ~covered & sh.intersects(shape, cells)
    values[cut] = sh.area(sh.intersection(cells[cut], shape))
    return values


def analyze(geo, settings, dc_results):
    """Return Celsius maps, component temperatures and a steady-state heat balance.

    grid_mm must divide material_mm. copper_um (in layer order or by name)
    and dielectric_mm (between adjacent foils) describe the stack. Uniform
    layer_power_w is distributed by board area. part_power_w is injected at a
    body coupled through contact_k_w (scalar or keyed by reference). The pad
    polygons are unioned per mounting layer. part_pad_ids can specify contact
    IDs explicitly; absent matching pads, the reported fallback uses the part
    bounds. dc_results is a list or mapping of simultaneous loss results with
    heat_cells. Each point represents a square of side dc_cell_mm. Convection
    references ambient_c. fixed_edges optionally pins named rectangular cut
    faces (left, right, top, bottom) to ambient. Geometry is never modified.
    Set method to "conformal" for interface-conforming triangular elements.
    """
    method = settings.get("method", "fixed")
    if method == "conformal":
        from .conformal import analyze as conforming_heat

        return conforming_heat(geo, settings, dc_results)
    if method != "fixed":
        raise ValueError(f"Unknown heat method: {method}")
    layers = list(geo["layers"])
    nz = len(layers)
    if nz < 2 or len(set(layers)) != nz:
        raise ValueError("At least two unique copper layers are required")
    pitch = _positive(settings.get("grid_mm", 1), "grid_mm")
    tile = _positive(settings.get("material_mm", 1), "material_mm")
    if round(tile / pitch) < 1 or not np.isclose(tile / pitch, round(tile / pitch)):
        raise ValueError("material_mm must be an integer multiple of grid_mm")
    board = _shape(geo["outline"])
    if board.is_empty or not board.is_valid or board.area <= 0:
        raise ValueError("Invalid board outline")
    xmin, ymin, xmax, ymax = board.bounds
    x, y = _edges(xmin, xmax, pitch), _edges(ymin, ymax, pitch)
    dx, dy = np.diff(x), np.diff(y)
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    cells = sh.box(x[None, :-1], y[:-1, None], x[None, 1:], y[1:, None])
    area = _areas(board, cells)
    valid = area > 0
    n = int(valid.sum())
    ids = np.full(cells.shape, -1, dtype=int)
    ids[valid] = np.arange(n)
    area_m2 = area[valid] * 1e-6
    tx, ty = _edges(xmin, xmax, tile), _edges(ymin, ymax, tile)
    tiles = sh.box(tx[None, :-1], ty[:-1, None], tx[None, 1:], ty[1:, None])
    tile_area = _areas(board, tiles)
    column = np.searchsorted(tx, xc, side="right") - 1
    row = np.searchsorted(ty, yc, side="right") - 1
    copper = settings.get("copper_um", [35] * nz)
    if isinstance(copper, dict):
        copper = [copper[layer] for layer in layers]
    cu = np.array([_positive(v, "copper_um") for v in copper]) * 1e-6
    gaps = (
        np.array(
            [
                _positive(v, "dielectric_mm")
                for v in settings.get("dielectric_mm", [1.5 / (nz - 1)] * (nz - 1))
            ]
        )
        * 1e-3
    )
    if len(cu) != nz or len(gaps) != nz - 1:
        raise ValueError("Stack dimensions do not match the layers")
    kcu = _positive(settings.get("copper_w_mk", 390), "copper_w_mk")
    kd = _positive(settings.get("dielectric_w_mk", 0.3), "dielectric_w_mk")
    plating = (
        _positive(settings.get("via_plating_um", 20), "via_plating_um", True) / 1000
    )
    depth = (np.r_[0, gaps] + np.r_[gaps, 0]) / 2
    net_data = geo["nets"].values()
    # Allocate the same physical plated area across the fixed material tiles
    # on every mesh. Blind vias only bridge gaps inside their physical span.
    plated = np.zeros((nz - 1, *tiles.shape))
    for net in net_data:
        for via in net.get("vias", []):
            if plating == 0:
                continue
            indices = [layers.index(name) for name in via["layers"]]
            if len(set(indices)) < 2:
                raise ValueError("A via requires its physical layer span")
            radius = _positive(via["drill_mm"], "drill_mm") / 2
            center = sh.Point(via["xy"])
            rim = center.buffer(radius + plating, quad_segs=32).difference(
                center.buffer(radius, quad_segs=32)
            )
            rim = rim.intersection(board)
            if rim.is_empty:
                continue
            a, b, c, d = rim.bounds
            ii = np.flatnonzero((tx[:-1] < c) & (tx[1:] > a))
            jj = np.flatnonzero((ty[:-1] < d) & (ty[1:] > b))
            where = np.ix_(jj, ii)
            occupied = _areas(rim, tiles[where])
            for z in range(min(indices), max(indices)):
                plated[z][where] += occupied

    left, right, conductance = [], [], []

    def connect(a, b, g):
        left.append(np.atleast_1d(a))
        right.append(np.atleast_1d(b))
        conductance.append(np.atleast_1d(g))

    fixed = set(settings.get("fixed_edges", []))
    if fixed - set(("left", "right", "top", "bottom")):
        raise ValueError("Unknown fixed edge")
    rectangle = board.equals(sh.box(*board.bounds))
    if fixed and not rectangle:
        raise ValueError("Fixed cut faces require a rectangular board outline")
    edge_g = np.zeros(nz * n)
    for z, layer in enumerate(layers):
        shapes = [r for net in net_data for r in net.get("copper", {}).get(layer, [])]
        metal = _shape(shapes).intersection(board)
        fraction = np.divide(
            _areas(metal, tiles),
            tile_area,
            out=np.zeros_like(tile_area),
            where=tile_area > 0,
        )
        sheet = kd * depth[z] + kcu * cu[z] * fraction[row[:, None], column[None, :]]
        for axis in (0, 1):
            # Run along x; transpose the geometry for y-directed links.
            s, i = (sheet, ids) if axis == 0 else (sheet.T, ids.T)
            width, height, along, across = (
                (dx, dy, x, y) if axis == 0 else (dy, dx, y, x)
            )
            ok = (i[:, :-1] >= 0) & (i[:, 1:] >= 0)
            j, k = np.nonzero(ok)
            span = height[j]
            if not rectangle and len(j):
                ab = np.stack((along[k + 1], across[j]), axis=1)
                cd = np.stack((along[k + 1], across[j + 1]), axis=1)
                coords = np.stack((ab, cd), axis=1)
                lines = sh.linestrings(coords if axis == 0 else coords[:, :, ::-1])
                span = sh.length(sh.intersection(lines, board))
            g = 2 * span / (width[k] / s[j, k] + width[k + 1] / s[j, k + 1])
            connect(z * n + i[j, k], z * n + i[j, k + 1], g)
            ends = ("left", "right") if axis == 0 else ("top", "bottom")
            for name, edge in zip(ends, (0, -1)):
                if name in fixed:
                    edge_g[z * n + i[:, edge]] += 2 * s[:, edge] * height / width[edge]
        if z < nz - 1:
            dz = gaps[z] + (cu[z] + cu[z + 1]) / 2
            pf = np.divide(
                plated[z], tile_area, out=np.zeros_like(tile_area), where=tile_area > 0
            )
            if pf.max() > 1 + 1e-9:
                raise ValueError("Overlapping via cross sections exceed tile area")
            f = pf[row[:, None], column[None, :]][valid]
            connect(
                z * n + np.arange(n),
                (z + 1) * n + np.arange(n),
                area_m2 * ((1 - f) * kd / gaps[z] + f * kcu / dz),
            )

    ambient = float(settings.get("ambient_c", 25))
    if not np.isfinite(ambient):
        raise ValueError("Non-finite ambient")
    parts = settings.get("part_power_w", {})
    size = nz * n + len(parts)
    air = np.zeros(size)
    air[:n] = area_m2 * _positive(
        settings.get("h_top_w_m2k", 6), "top convection", True
    )
    air[(nz - 1) * n : nz * n] = area_m2 * _positive(
        settings.get("h_bottom_w_m2k", 6), "bottom convection", True
    )
    power = np.zeros(size)
    for layer, watts in settings.get("layer_power_w", {}).items():
        z = layers.index(layer)
        power[z * n : (z + 1) * n] += (
            _positive(watts, "layer power", True) * area_m2 / area_m2.sum()
        )

    def weights(shape):
        shape = shape.intersection(board)
        if shape.is_empty or shape.area <= 0:
            raise ValueError("Source has no positive area on the board")
        a, b, c, d = shape.bounds
        ii = np.flatnonzero((x[:-1] < c) & (x[1:] > a))
        jj = np.flatnonzero((y[:-1] < d) & (y[1:] > b))
        where = np.ix_(jj, ii)
        occupied = _areas(shape, cells[where])
        mask = occupied > 0
        return ids[where][mask], occupied[mask] / occupied.sum()

    side = _positive(settings.get("dc_cell_mm", 1), "dc_cell_mm") / 2
    results = dc_results.values() if isinstance(dc_results, dict) else dc_results
    for entry in results:
        for cell in entry["heat_cells"]:
            xx, yy = cell["x_mm"], cell["y_mm"]
            local, weight = weights(sh.box(xx - side, yy - side, xx + side, yy + side))
            power[layers.index(cell["layer"]) * n + local] += (
                _positive(cell["watts"], "DC power", True) * weight
            )

    contacts = {}
    for number, (ref, watts) in enumerate(parts.items(), nz * n):
        part = geo["parts"][ref]
        explicit = settings.get("part_pad_ids", {}).get(ref)
        polygons = [
            poly
            for net in net_data
            for pin, pads in net.get("pads", {}).items()
            if (pin in explicit if explicit is not None else pin.startswith(ref + "."))
            for pad in pads
            if pad["layer"] == part["layer"]
            for poly in pad["polygons"]
        ]
        if explicit is not None and not polygons:
            raise ValueError(f"No matching contact polygons for {ref}")
        shape = _shape(polygons) if polygons else sh.box(*part["bounds"])
        contacts[ref] = "pad union" if polygons else "footprint bounds"
        local, weight = weights(shape)
        rc = settings.get("contact_k_w", 12)
        resistance = _positive(
            rc[ref] if isinstance(rc, dict) else rc, "contact resistance"
        )
        connect(
            np.full(len(weight), number),
            n * layers.index(part["layer"]) + local,
            weight / resistance,
        )
        power[number] = _positive(watts, "part power", True)
        if ref in settings.get("part_to_air_k_w", {}):
            air[number] = 1 / _positive(
                settings["part_to_air_k_w"][ref], "part to air resistance"
            )

    a, b, g = map(np.concatenate, (left, right, conductance))
    sinks = air.copy()
    sinks[: nz * n] += edge_g
    # Every connected island needs an absolute temperature reference.
    positive = g > 0
    adj = sparse.coo_matrix(
        (g[positive], (a[positive], b[positive])), shape=(size, size)
    ).tocsr()
    groups, labels = sparse.csgraph.connected_components(adj, directed=False)
    if np.any(np.bincount(labels, weights=sinks, minlength=groups) <= 0):
        raise ValueError("A thermal island has no heat sink")
    matrix = -(adj + adj.T)
    matrix.setdiag(sinks - np.asarray(matrix.sum(axis=1)).ravel())
    if size <= 25000:
        rise = spsolve(matrix, power)
    else:
        pre = sparse.diags(1 / matrix.diagonal())
        rise, status = cg(matrix, power, M=pre, rtol=1e-10, atol=0, maxiter=5000)
        if status:
            raise ArithmeticError(f"Heat solver did not converge ({status})")
    imbalance = abs(float(np.dot(sinks, rise) - power.sum()))
    if not np.isfinite(rise).all() or imbalance > 1e-7 * max(power.sum(), 1e-9):
        raise ArithmeticError("Heat balance failed")
    yy, xx = np.nonzero(valid)
    maps = [
        {
            "name": "Board temperature",
            "layer": layer,
            "unit": "degC",
            "points": np.column_stack(
                (xc[xx], yc[yy], ambient + rise[z * n : (z + 1) * n])
            ).tolist(),
        }
        for z, layer in enumerate(layers)
    ]
    return dict(
        maps=maps,
        peak_board_c=float(ambient + rise[: nz * n].max()),
        body_c={ref: float(ambient + rise[nz * n + i]) for i, ref in enumerate(parts)},
        contact_model=contacts,
        power_w=float(power.sum()),
        convected_w=float(np.dot(air, rise)),
        cut_w=float(np.dot(edge_g, rise[: nz * n])),
        balance_error_w=imbalance,
        grid_mm=pitch,
        material_mm=tile,
    )
