"""Cut-face finite-volume copper networks with explicit plated interconnects."""

import math
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

MU = 4e-7 * math.pi


def polygon(records):
    geometries = [
        Polygon(item["shell"], item.get("holes", []))
        for item in records
        if len(item["shell"]) >= 3
    ]
    return unary_union([g if g.is_valid else g.buffer(0) for g in geometries])


def slab(frequency, thickness, resistivity):
    """Complex internal impedance per square for symmetric conduction [SI]."""
    if frequency == 0:
        return resistivity / thickness
    q = np.sqrt(1j * 2 * math.pi * frequency * MU / resistivity) * thickness / 2
    return resistivity / thickness * q / np.tanh(q)


def line_parts(geom):
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif hasattr(geom, "geoms"):
        for item in geom.geoms:
            yield from line_parts(item)


class Sheet:
    def __init__(self, geometry, material, net, mesh_mm, bounds=None):
        if mesh_mm <= 0:
            raise ValueError("Positive mesh pitch required")
        self.net = geometry["nets"][net]
        self.layers = geometry["layers"]
        self.mat = material
        self.h = mesh_mm
        self.rho = material.get("rho_ohm_m", 1.7241e-8)
        if self.rho <= 0:
            raise ValueError("Positive resistivity required")
        self.edges, self.where, lookup = [], [], {}

        # Nodes sit on a globally anchored grid so overlay coordinates do not depend on the net.
        def node(layer, i, j):
            key = (layer, i, j)
            if key not in lookup:
                lookup[key] = len(self.where)
                self.where.append((layer, (i + 0.5) * self.h, (j + 0.5) * self.h))
            return lookup[key]

        clip = box(*bounds) if bounds else None
        for l in self.layers:
            copper = polygon(self.net["copper"].get(l, []))
            if clip is not None:
                copper = copper.intersection(clip)
            if copper.is_empty:
                continue
            ext = np.array(copper.bounds)
            for axis in (0, 1):
                other = 1 - axis
                for i in range(
                    math.ceil(ext[axis] / self.h),
                    math.floor(ext[axis + 2] / self.h) + 1,
                ):
                    endpoints = [
                        [i * self.h, ext[other] - 0.1],
                        [i * self.h, ext[other + 2] + 0.1],
                    ]
                    if axis:
                        endpoints = [p[::-1] for p in endpoints]
                    line = LineString(endpoints)
                    cut = copper.intersection(line).difference(copper.boundary)
                    for part in line_parts(cut):
                        low, high = sorted(
                            [part.coords[0][other], part.coords[-1][other]]
                        )
                        for j in range(
                            math.floor(low / self.h), math.ceil(high / self.h)
                        ):
                            width = min(high, (j + 1) * self.h) - max(low, j * self.h)
                            if width <= 1e-8:
                                continue
                            left = node(l, i - 1, j) if not axis else node(l, j, i - 1)
                            right = node(l, i, j) if not axis else node(l, j, i)
                            self.edges.append((left, right, self.h, width, l, False))
        coord = np.asarray([(x, y) for _, x, y in self.where])
        if not len(coord):
            raise ValueError("No resolved metal")
        levels = np.array([l for l, _, _ in self.where])
        for via in self.net.get("vias", []):
            x, y = via["xy"]
            if bounds and not clip.covers(Point(x, y)):
                continue
            # One ring node per flashed layer. Radial spreading to the wall uses actual land radius.
            radial = np.hypot(coord[:, 0] - x, coord[:, 1] - y)
            prev = None
            for l in self.layers:
                if l not in via["layers"]:
                    continue
                sel = np.flatnonzero(
                    (levels == l) & (radial <= via["land_mm"] / 2 + 0.72 * self.h)
                )
                if not len(sel):
                    continue
                nearest = radial[sel].min()
                sel = sel[
                    radial[sel]
                    <= max(nearest + 0.2 * self.h, via["drill_mm"] / 2 + 0.45 * self.h)
                ]
                hub = len(self.where)
                self.where.append((l, x, y))
                for n in sel:
                    self.edges.append(
                        (
                            n,
                            hub,
                            max(self.h * 0.12, radial[n] - via["drill_mm"] / 2),
                            math.pi * via["drill_mm"] / len(sel),
                            l,
                            False,
                        )
                    )
                if prev is not None:
                    lower, index = prev
                    height = abs(material["depth_mm"][l] - material["depth_mm"][lower])
                    self.edges.append(
                        (index, hub, height, math.pi * via["drill_mm"], l, True)
                    )
                prev = (l, hub)
        self.xy = np.array([(x, y) for _, x, y in self.where])
        self.levels = np.array([l for l, _, _ in self.where])
        self.a, self.b = np.array([(v[0], v[1]) for v in self.edges], dtype=int).T
        self.length = np.array([e[2] for e in self.edges])
        self.width = np.array([e[3] for e in self.edges])
        self.thick = np.array(
            [
                material.get("plating_mm", 0.025)
                if e[5]
                else material["copper_mm"][e[4]]
                for e in self.edges
            ]
        )

    def electrode(self, spec):
        # Terminals may be described by a named pin or an arbitrary disk on a particular layer.
        if isinstance(spec, str):
            pads = self.net["pads"][spec]
            regions = [(p["layer"], polygon(p["polygons"])) for p in pads]
        else:
            regions = [
                (
                    spec["layer"],
                    Point(*spec["xy"]).buffer(spec.get("radius_mm", self.h * 0.65)),
                )
            ]
        nodes = set()
        for layer, region in regions:
            if region.is_empty:
                continue
            allowed = np.flatnonzero(self.levels == layer)
            if not len(allowed):
                continue
            inside = contains_xy(
                region.buffer(1e-8), self.xy[allowed, 0], self.xy[allowed, 1]
            )
            nodes.update(allowed[inside])
            if not inside.any():
                pos = region.representative_point()
                distance = np.linalg.norm(self.xy[allowed] - [pos.x, pos.y], axis=1)
                if min(distance) <= self.h * 0.76:
                    nodes.add(allowed[distance.argmin()])
        if not nodes:
            raise ValueError("Unresolved terminal; reduce mesh pitch: " + str(spec))
        return sorted(nodes)

    def solve(self, ports, freq=0.0, impressed=None, reference_height_mm=0.0):
        if freq < 0 or reference_height_mm < 0:
            raise ValueError("Invalid frequency or spacing")
        current = np.array([float(p["current_A"]) for p in ports])
        if len(ports) < 2 or abs(current.sum()) > 1e-10 or max(current) <= 0:
            raise ValueError("Balanced nonzero terminal currents required")
        ids = np.arange(len(self.where))
        used = set()
        terminals = []
        for p in ports:
            group = sorted(
                set(n for contact in p["contacts"] for n in self.electrode(contact))
            )
            if used.intersection(group):
                raise ValueError("Electrodes overlap at this mesh pitch")
            used.update(group)
            ids[group] = group[0]
            terminals.append(group[0])
        _, nodes = np.unique(ids, return_inverse=True)
        a, b = nodes[self.a], nodes[self.b]
        keep = a != b
        a, b = a[keep], b[keep]
        term = nodes[terminals]
        N = int(nodes.max()) + 1
        t = self.thick[keep] * 0.001
        unit = np.array([slab(freq, u, self.rho) for u in np.unique(t)])
        z = unit[np.searchsorted(np.unique(t), t)]
        z = z * self.length[keep] / self.width[keep]
        via = np.array([e[5] for e in self.edges])[keep]
        if freq and reference_height_mm:
            z += (
                1j
                * 2
                * np.pi
                * freq
                * MU
                * reference_height_mm
                * 0.001
                * (~via)
                * self.length[keep]
                / self.width[keep]
            )
        adm = 1 / z
        graph = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(N, N)).tocsr()
        _, tags = connected_components(graph, directed=False)
        if np.any(tags[term] != tags[term[0]]):
            raise ValueError(
                "Disconnected contacts; check geometry and grid convergence"
            )
        # Excited floating islands need their own gauge, including a closed conductor with no ports.
        participating = (
            np.ones(N, dtype=bool) if impressed is not None else tags == tags[term[0]]
        )
        free = participating.copy()
        for group in np.unique(tags[participating]):
            gauge = (
                term[-1]
                if group == tags[term[-1]]
                else np.flatnonzero(tags == group)[0]
            )
            free[gauge] = False
        K = sparse.coo_matrix(
            (np.r_[adm, adm, -adm, -adm], (np.r_[a, b, a, b], np.r_[a, b, b, a])),
            shape=(N, N),
        ).tocsr()
        volts = np.zeros(N, dtype=z.dtype)
        source = (
            np.zeros(len(a), dtype=z.dtype)
            if impressed is None
            else np.asarray(impressed)[keep]
        )
        rhs = np.zeros_like(volts)
        np.add.at(rhs, term, current)
        np.add.at(rhs, a, -adm * source)
        np.add.at(rhs, b, adm * source)
        volts[free] = spsolve(K[free][:, free].tocsc(), rhs[free])
        flows = (volts[a] - volts[b] + source) * adm
        flows[~participating[a]] = 0
        allflows = np.zeros(len(keep), dtype=z.dtype)
        allflows[keep] = flows
        watts = np.zeros(len(keep))
        watts[keep] = abs(flows) ** 2 * z.real
        residual = np.zeros(N, dtype=z.dtype)
        np.add.at(residual, a, flows)
        np.add.at(residual, b, -flows)
        np.add.at(residual, term, -current)
        return (
            volts[nodes],
            volts[term],
            allflows,
            watts,
            float(max(abs(residual[participating]))),
        )

    def positions_m(self):
        return (
            np.column_stack((self.xy, [self.mat["depth_mm"][l] for l in self.levels]))
            * 0.001
        )
