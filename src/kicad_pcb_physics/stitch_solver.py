"""Greedy, two-terminal stitching studies on the extracted finite-volume Sheet.

All candidate coordinates are in mm. Sheet electrode names or disks are accepted.
Each candidate is a series of plated spans with a finite access impedance to the
resolved annular cells on each contacted layer. The foil and its cell boundaries stay fixed:
this is a subcell insertion model, including neither drill removal nor repour.
Absolute open-circuit results are serialized as null. Gains are before minus
after. Each stage exhausts the remaining single additions to this model.

At each frequency, factor the grounded sheet once. For unit-current port
voltage v and oriented barrel incidence B, the connected-case port change is
-(B.T v).T (Z_barrel + B.T K^-1 B)^-1 (B.T v). Transposes are bilinear
(reciprocity), not Hermitian. Cache only on-site transfer blocks and columns
for winners. Independent island gauges enforce zero net floating current.
Magnetic coenergy and reciprocity follow Haus and Melcher, Electromagnetic
Fields and Energy, section 11.4 (https://web.mit.edu/6.013_book/www/chapter11/11.4.html).
"""

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box

from . import fields
from .mesh import MU, Sheet, polygon, slab
from .studies import mapset


def _number(value, name, positive=False):
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {name}")
    return value


def _edge_z(f, rho, thickness, squares, external_h):
    thickness = np.asarray(thickness)
    values, index = np.unique(thickness, return_inverse=True)
    internal = np.asarray([slab(f, t * 0.001, rho) for t in values])[index]
    return internal * squares + (2j * np.pi * f * external_h if f else 0)


def _barrel_external(length, drill, radius):
    # Coaxial return boundary, only when explicitly specified by the caller.
    if radius is None:
        return np.zeros_like(length)
    if np.any(radius <= drill / 2):
        raise ValueError("via_return_radius_mm must exceed each barrel radius")
    return MU / (2 * np.pi) * length * 0.001 * np.log(radius / (drill / 2))


@dataclass
class Site:
    xy: tuple
    nodes: tuple
    layers: tuple
    access_squares: np.ndarray
    rims: np.ndarray
    heights: np.ndarray
    spoke_mm: np.ndarray
    span: slice


def _sheet(g, material, spec):
    bounds = spec.get("bounds_mm")
    if bounds is None:
        return Sheet(g, material, spec["net"], spec["mesh_mm"])
    window = box(*bounds)
    original = g["nets"][spec["net"]]
    copper = {}
    for layer, records in original["copper"].items():
        clipped = polygon(records).intersection(window)
        # Touching the window with only a line or point adds no sheet area.
        # GEOS defines no boundary for a mixed-dimensional GeometryCollection.
        pieces = list(shapely.get_parts(clipped))
        polygons = []
        while pieces:
            item = pieces.pop()
            if item.geom_type == "Polygon" and not item.is_empty:
                polygons.append(
                    {
                        "shell": list(item.exterior.coords),
                        "holes": [list(r.coords) for r in item.interiors],
                    }
                )
            elif item.geom_type in ("GeometryCollection", "MultiPolygon"):
                pieces.extend(item.geoms)
        copper[layer] = polygons
    net = {
        **original,
        "copper": copper,
        "vias": [v for v in original.get("vias", []) if window.covers(Point(v["xy"]))],
    }
    return Sheet(
        {**g, "nets": {spec["net"]: net}}, material, spec["net"], spec["mesh_mm"]
    )


def _sites(sheet, coords, spec, drill, land):
    chosen_layers = spec.get("layers", sheet.layers)
    if len(set(chosen_layers)) != len(chosen_layers) or set(chosen_layers) - set(
        sheet.layers
    ):
        raise ValueError("Unknown or duplicate contact layer")
    layers = sorted(chosen_layers, key=sheet.mat["depth_mm"].__getitem__)
    h = sheet.h
    grid = np.all(np.isclose(sheet.xy / h % 1, 0.5, atol=1e-7, rtol=0), axis=1)
    layers_data = []
    for layer in layers:
        original = np.flatnonzero(grid & (sheet.levels == layer))
        if not original.size:
            continue
        # First occurrence of a grid coordinate is always a foil node.
        _, first = np.unique(sheet.xy[original], axis=0, return_index=True)
        original = original[first]
        metal = polygon(sheet.net["copper"].get(layer, []))
        shapely.prepare(metal)
        layers_data.append((layer, original, cKDTree(sheet.xy[original]), metal))
    sites, skipped = [], []
    start = 0
    for xy in coords:
        land_shape = Point(xy).buffer(land / 2)
        annulus = land_shape.difference(Point(xy).buffer(drill / 2))
        contacts, access, hit, rims, spoke_mm = [], [], [], [], []
        for layer, original, tree, metal in layers_data:
            local = tree.query_ball_point(xy, land / 2 + h / math.sqrt(2))
            if not local or not metal.intersects(land_shape):
                continue
            metal_land = (
                annulus
                if metal.covers(land_shape)
                else shapely.clip_by_rect(metal, *land_shape.bounds).intersection(
                    annulus
                )
            )
            if metal_land.is_empty:
                continue
            points = original[local]
            centers = sheet.xy[points]
            cells = shapely.box(
                centers[:, 0] - h / 2,
                centers[:, 1] - h / 2,
                centers[:, 0] + h / 2,
                centers[:, 1] + h / 2,
            )
            area = shapely.area(shapely.intersection(cells, metal_land))
            points = np.sort(points[area > 0])
            if not len(points):
                continue
            r = np.linalg.norm(sheet.xy[points] - xy, axis=1)
            keep = r <= max(r.min() + 0.2 * h, drill / 2 + 0.45 * h)
            contacts.extend(points[keep])
            hit.append(layer)
            rims.append(len(contacts) - 1)
            lengths = np.maximum(0.12 * h, r[keep] - drill / 2)
            spoke_mm.extend(lengths)
            access.extend(lengths * np.count_nonzero(keep) / (np.pi * drill))
        if len(hit) < 2:
            skipped.append(
                {"xy": list(xy), "reason": "Fewer than two resolved copper contacts"}
            )
            continue
        heights = np.diff([sheet.mat["depth_mm"][l] for l in hit])
        if np.any(heights <= 0):
            raise ValueError("Contact layers must have distinct depths")
        stop = start + len(contacts) - 1
        sites.append(
            Site(
                xy,
                tuple(contacts),
                tuple(hit),
                np.array(access),
                np.asarray(rims[:-1], int),
                heights,
                np.asarray(spoke_mm),
                slice(start, stop),
            )
        )
        start = stop
    return sites, skipped


def _signal_shape(g, spec):
    path = spec.get("exciting_path_mm")
    if path is not None:
        p = np.asarray(path, float)
        if (
            p.ndim != 2
            or p.shape[1] != 3
            or len(p) < 2
            or not np.isfinite(p).all()
            or np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) == 0)
        ):
            raise ValueError("exciting_path_mm needs distinct consecutive xyz points")
        return LineString(p[:, :2]), p
    signal = spec.get("signal_net")
    if signal is None:
        return None, None
    traces = []
    for tr in g["nets"][signal]["tracks"]:
        a, b = np.array(tr["a"]), np.array(tr["b"])
        points = [a, b]
        if "mid" in tr:
            m = np.array(tr["mid"])
            u, v = m - a, b - a
            center = np.linalg.solve(2 * np.stack([u, v]), [u @ u, v @ v])
            start, mid, stop = [math.atan2(*(p - a - center)[::-1]) for p in (a, m, b)]
            sweep = (stop - start) % math.tau
            if (mid - start) % math.tau > sweep:
                sweep -= math.tau
            t = np.linspace(
                start,
                start + sweep,
                1
                + max(
                    2,
                    math.ceil(
                        abs(sweep) * np.linalg.norm(center) / (spec["mesh_mm"] / 2)
                    ),
                ),
            )
            points = (
                a
                + center
                + np.linalg.norm(center) * np.column_stack((np.cos(t), np.sin(t)))
            )
        traces.append(LineString(points))
    if not traces:
        raise ValueError("signal_net has no projected tracks")
    return shapely.union_all(traces), None


class MagneticModes:
    """Galerkin projection of regularized Neumann magnetic energy.

    The forward filament is in the basis exactly. The return chord and optional
    additional probe paths sample independent return fields. Their mutual Gram
    matrix defines a common inner product for all foil spokes, axial spans and
    foil edges. Omitted orthogonal return fields limit accuracy; optional local
    leakage terms and refinement of the probe set are explicit assumptions.
    """

    def __init__(self, sheet, source, sink, path, spec):
        xyz = sheet.positions_m() * 1000
        src, dst = [xyz[group].mean(axis=0) for group in (source, sink)]
        self.radius = (
            _number(spec.get("wire_radius_mm", 0.08), "wire_radius_mm", True) * 0.001
        )
        step = _number(spec.get("filament_step_mm", 0.4), "filament_step_mm", True)
        curves = [
            np.vstack((dst, path, src)),
            np.array([src, dst]),
            *spec.get("return_modes_mm", []),
        ]
        self.filaments = []
        for curve in curves:
            curve = np.asarray(curve, float)
            if (
                curve.ndim != 2
                or curve.shape[1] != 3
                or len(curve) < 2
                or not np.isfinite(curve).all()
            ):
                raise ValueError("Magnetic basis paths must be finite xyz polylines")
            curve = curve[
                np.r_[True, np.linalg.norm(np.diff(curve, axis=0), axis=1) > 1e-10]
            ]
            if len(curve) < 2:
                continue
            self.filaments.append(fields.filaments(curve * 0.001, 1.0, step * 0.001))
        gram = np.array(
            [
                self.coupling(a, b, quadrature=6).sum(axis=0)
                for a, b, _ in self.filaments
            ]
        )
        gram = (gram + gram.T) / 2
        eigen, vectors = np.linalg.eigh(gram)
        if eigen[-1] <= 0 or eigen[0] < -1e-8 * eigen[-1]:
            raise ValueError("Non-passive filament basis; refine filament_step_mm")
        keep = eigen > 1e-9 * eigen[-1]
        self.normalize = vectors[:, keep] / np.sqrt(eigen[keep])
        self.forward = gram[0] @ self.normalize

    def coupling(self, a, b, quadrature=2):
        direction = b - a
        points, weights = np.polynomial.legendre.leggauss(quadrature)
        result = np.zeros((len(a), len(self.filaments)))
        for j, wire in enumerate(self.filaments):
            for u, weight in zip(points, weights):
                vector, _ = fields.potentials(
                    a + 0.5 * (1 + u) * direction, *wire, self.radius
                )
                result[:, j] += 0.5 * weight * np.einsum("ij,ij->i", vector, direction)
        return result

    def linkage(self, a, b):
        return self.coupling(a, b) @ self.normalize


def _passive_loop(ports, omega):
    """Reject an invalid two-port even if its summed series reactance is positive."""
    if not np.isfinite(ports).all():
        raise ArithmeticError("Non-finite magnetic port impedance")
    if abs(ports[1, 0] - ports[0, 1]) > 1e-7 * np.linalg.norm(ports):
        raise ArithmeticError("Magnetic port coupling violates reciprocity")
    ports = (ports + ports.T) / 2
    loss = ports.real
    energy = ports.imag / omega
    resistance = float(loss.sum())
    inductance = float(energy.sum())
    min_r = float(np.linalg.eigvalsh(loss)[0])
    min_l = float(np.linalg.eigvalsh(energy)[0])
    certificate = {"minimum_R_ohm": min_r, "minimum_L_H": min_l}
    if (
        inductance < 0
        or resistance < 0
        or min_r < -1e-10 * np.linalg.norm(loss)
        or min_l < -1e-10 * np.linalg.norm(energy)
    ):
        return None, "Rejected non-passive reduced two-port", certificate
    return inductance, "Passive at the evaluated frequency", certificate


class Network:
    def __init__(self, sheet, sites, spec, f, path, distance):
        self.sheet, self.sites, self.f, self.distance = sheet, sites, f, distance
        source, sink = (sheet.electrode(spec[k]) for k in ("source", "sink"))
        if set(source) & set(sink):
            raise ValueError("Source and sink overlap on this mesh")
        ids = np.arange(len(sheet.xy))
        for group in (source, sink):
            ids[group] = group[0]
        _, nodes = np.unique(ids, return_inverse=True)
        self.nodes = nodes
        term = nodes[[source[0], sink[0]]]
        self.term = term
        a, b = nodes[sheet.a], nodes[sheet.b]
        via = np.array([e[5] for e in sheet.edges])
        spacing = spec.get("return_spacing_mm", 0)
        radius = spec.get("via_return_radius_mm")
        ext = MU * spacing * 0.001 * (~via) * sheet.length / sheet.width
        ext[via] = _barrel_external(sheet.length[via], sheet.width[via] / np.pi, radius)
        self.z = _edge_z(f, sheet.rho, sheet.thick, sheet.length / sheet.width, ext)
        adm = 1 / self.z
        n = nodes.max() + 1
        matrix = sparse.coo_matrix(
            (np.r_[adm, adm, -adm, -adm], (np.r_[a, b, a, b], np.r_[a, b, b, a])),
            shape=(n, n),
        ).tocsc()
        _, self.tags = connected_components(matrix, directed=False)
        _, gauges = np.unique(self.tags, return_index=True)
        gauges[self.tags[term[1]]] = term[1]
        self.free = np.ones(n, bool)
        self.free[gauges] = False
        self.lu = splu(matrix[self.free][:, self.free]) if self.free.any() else None
        rhs = np.zeros(n, self.z.dtype)
        rhs[term] = [1, -1]
        self.component_drive = np.bincount(self.tags, weights=rhs.real)
        self.omega = math.tau * f
        self.modes = (
            MagneticModes(sheet, source, sink, path, spec)
            if path is not None and f
            else None
        )
        self.external = self.modes is not None
        nm = len(self.modes.forward) if self.external else 0
        self.forward_h = (
            float(self.modes.forward @ self.modes.forward) if self.external else None
        )
        self.emf = np.empty((len(adm), nm), self.z.dtype)
        correction = np.zeros((len(nodes), nm))
        rhs = np.column_stack((rhs, np.zeros((n, nm), self.z.dtype)))
        xyz = sheet.positions_m()
        if self.external:
            for group in (source, sink):
                center = xyz[group].mean(axis=0)
                correction[group] = self.modes.linkage(
                    np.broadcast_to(center, (len(group), 3)), xyz[group]
                )
            self.emf = (
                -1j
                * self.omega
                * (
                    self.modes.linkage(xyz[sheet.a], xyz[sheet.b])
                    + correction[sheet.a]
                    - correction[sheet.b]
                )
            )
            np.add.at(rhs[:, 1:], a, -self.emf * adm[:, None])
            np.add.at(rhs[:, 1:], b, self.emf * adm[:, None])
        self.solutions = self.solve(rhs)
        flow = self.solutions[a] - self.solutions[b]
        flow[:, 1:] += self.emf
        flow *= adm[:, None]
        self.base_ports = np.vstack(
            (self.solutions[term[0]] - self.solutions[term[1]], -self.emf.T @ flow)
        )
        aa, bb, blocks = [], [], []
        radial_from, radial_to = [], []
        axial_from, axial_to = [], []
        for site in sites:
            contact = nodes[list(site.nodes)]
            aa.extend(contact[:-1])
            bb.extend(contact[1:])
            drill = spec["drill_mm"]
            d = np.diff(np.eye(len(contact)), axis=0)
            barrel = _edge_z(
                f,
                sheet.rho,
                np.full_like(site.heights, spec["plating_mm"]),
                site.heights / (np.pi * drill),
                _barrel_external(site.heights, drill, radius),
            )
            access = _edge_z(
                f,
                sheet.rho,
                [sheet.mat["copper_mm"][l] for l in sheet.levels[list(site.nodes)]],
                site.access_squares,
                MU * spacing * 0.001 * site.access_squares,
            )
            block = (d * access) @ d.T
            block[site.rims, site.rims] += barrel
            blocks.append(block)
            if self.external:
                foil = xyz[list(site.nodes)]
                wall = foil.copy()
                wall[:, :2] = np.asarray(site.xy) * 0.001
                radial_from.extend(foil)
                radial_to.extend(wall)
                axial_from.extend(wall[:-1])
                axial_to.extend(wall[1:])
        self.aa, self.bb = np.asarray(aa, int), np.asarray(bb, int)
        self.blocks = blocks
        self.t = self.solutions[self.aa] - self.solutions[self.bb]
        if self.external and sites:
            radial = self.modes.linkage(np.asarray(radial_from), np.asarray(radial_to))
            axial = self.modes.linkage(np.asarray(axial_from), np.asarray(axial_to))
            offset = 0
            for site in sites:
                count = len(site.nodes)
                spokes = radial[offset : offset + count] + correction[list(site.nodes)]
                self.t[site.span, 1:] -= (
                    1j * self.omega * (axial[site.span] + spokes[:-1] - spokes[1:])
                )
                offset += count
        self.diagonal = {}
        self.cross = {}

    def solve(self, rhs):
        out = np.zeros_like(rhs)
        if self.lu:
            out[self.free] = self.lu.solve(rhs[self.free])
        return out

    def columns(self, cols):
        rhs = np.zeros((len(self.free), len(cols)), dtype=self.z.dtype, order="F")
        idx = np.arange(len(cols))
        np.add.at(rhs, (self.aa[cols], idx), 1)
        np.add.at(rhs, (self.bb[cols], idx), -1)
        return self.solve(rhs)

    def prepare(self):
        # Batch spatial solves and share exact response blocks for subcell sites
        # using the same foil nodes. No square all-candidate matrix is stored.
        groups = {}
        for k, s in enumerate(self.sites):
            key = tuple(self.nodes[list(s.nodes)])
            groups.setdefault(key, []).append(k)
        batch = max(
            1, min(24, 12_000_000 // max(1, len(self.free) * self.z.itemsize * 3))
        )
        pending, cols = [], []

        def flush():
            response = self.columns(cols)
            for group, segment in pending:
                span = self.sites[group[0]].span
                local = (
                    response[self.aa[span], segment] - response[self.bb[span], segment]
                )
                for k in group:
                    self.diagonal[k] = self.blocks[k] + local
            cols.clear()
            pending.clear()

        for group in groups.values():
            span = self.sites[group[0]].span
            size = span.stop - span.start
            if cols and len(cols) + size > batch:
                flush()
            pending.append((group, slice(len(cols), len(cols) + size)))
            cols.extend(range(span.start, span.stop))
        if cols:
            flush()

    def accept(self, k):
        span = self.sites[k].span
        response = self.columns(np.arange(span.start, span.stop))
        self.cross[k] = response[self.aa] - response[self.bb]
        if k not in self.diagonal:
            self.diagonal[k] = self.blocks[k] + self.cross[k][span]

    def evaluate(self, selected, state=False):
        # Restore each floating gauge by component-wise current conservation.
        spans = [self.sites[k].span for k in selected]
        idx = (
            np.concatenate([np.arange(s.start, s.stop) for s in spans]).astype(int)
            if spans
            else np.empty(0, int)
        )
        u, v = self.tags[self.aa[idx]], self.tags[self.bb[idx]]
        tags = np.unique(np.r_[u, v, self.tags[self.term]])
        local = np.searchsorted(tags, self.tags[self.term])
        ui, vi = np.searchsorted(tags, u), np.searchsorted(tags, v)
        graph = sparse.coo_matrix(
            (np.ones(len(idx)), (ui, vi)), shape=(len(tags), len(tags))
        )
        _, groups = connected_components(graph, directed=False)
        if groups[local[0]] != groups[local[1]]:
            return None
        gauge = np.zeros((len(tags), self.base_ports.shape[0]), self.z.dtype)
        currents = np.zeros((len(idx), self.base_ports.shape[0]), self.z.dtype)
        if len(idx):
            ends = np.cumsum([0] + [s.stop - s.start for s in spans])
            K = np.empty((len(idx), len(idx)), self.z.dtype)
            for j, (k, s) in enumerate(zip(selected, spans)):
                cut = slice(ends[j], ends[j + 1])
                if j:
                    other = np.hstack([self.cross[prev][s] for prev in selected[:j]])
                    K[cut, : ends[j]], K[: ends[j], cut] = other, other.T
                K[cut, cut] = self.diagonal[k]
            _, refs = np.unique(groups, return_index=True)
            refs[groups[local[1]]] = local[1]
            mask = np.ones(len(tags), bool)
            mask[refs] = False
            C = np.zeros((len(tags), len(idx)))
            np.add.at(C, (ui, np.arange(len(idx))), 1)
            np.add.at(C, (vi, np.arange(len(idx))), -1)
            C = C[mask]
            drive = self.component_drive[tags[mask]]
            block = np.block([[K, C.T], [C, np.zeros((len(C), len(C)))]])
            demand = np.zeros((len(C), self.base_ports.shape[0]), self.z.dtype)
            demand[:, 0] = drive
            force = np.vstack((self.t[idx], demand))
            sol = np.linalg.solve(block, force)
            currents = sol[: len(idx)]
            gauge[mask] = -sol[len(idx) :]
            ports = self.base_ports - force.T @ sol
        else:
            ports = self.base_ports
        result = {
            "z": ports[0, 0],
            "v": ports[0, 0],
            "loop_H": None,
            "loop_status": "No directed forward path",
        }
        y = np.ones(1, self.z.dtype)
        if self.external:
            # a is the total projected flux amplitude: U.T i + forward * If.
            s = 1j * self.omega
            reaction = np.eye(len(self.modes.forward)) - ports[1:, 1:] / s
            ar, af = np.linalg.solve(
                reaction, np.column_stack((ports[1:, 0] / s, self.modes.forward))
            ).T
            zr = ports[0, 0] + ports[0, 1:] @ ar
            zrf = ports[0, 1:] @ af
            zfr = s * self.modes.forward @ ar
            zf = s * self.modes.forward @ af
            zport = np.array([[zf, zfr], [zrf, zr]])
            loop, status, certificate = _passive_loop(zport, self.omega)
            result.update(
                z=zr,
                v=zr + zrf,
                loop_H=loop,
                loop_status=status,
                loop_passivity=certificate,
            )
            y = np.r_[1, ar + af]
        if state:
            applied = currents @ y
            inj = np.zeros(len(self.free), self.z.dtype)
            np.add.at(inj, self.aa[idx], applied)
            np.add.at(inj, self.bb[idx], -applied)
            offsets = np.zeros_like(self.component_drive, dtype=self.z.dtype)
            offsets[tags] = gauge @ y
            volts = (self.solutions @ y - self.solve(inj) + offsets[self.tags])[
                self.nodes
            ]
            flow = (
                volts[self.sheet.a] - volts[self.sheet.b] + self.emf @ y[1:]
            ) / self.z
            active = np.isin(self.tags[self.nodes], tags[groups == groups[local[0]]])
            span_currents = np.split(
                applied, np.cumsum([span.stop - span.start for span in spans])[:-1]
            )
            barrel_currents = [
                q[self.sites[k].rims] for k, q in zip(selected, span_currents)
            ]
            result.update(
                volts=volts,
                flow=flow,
                active=active,
                barrels=np.concatenate(barrel_currents)
                if barrel_currents
                else np.empty(0),
            )
            if self.external and result["loop_H"] is not None:
                return_q = np.vdot(flow, self.z * flow) + sum(
                    np.vdot(q, self.blocks[k] @ q)
                    for k, q in zip(selected, span_currents)
                )
                return_modes = y[1:] - self.modes.forward
                result["energy_terms_H"] = {
                    "forward_self": self.forward_h,
                    "return_projected": float(np.vdot(return_modes, return_modes).real),
                    "twice_mutual": float(2 * (self.modes.forward @ return_modes).real),
                    "local_return": float(return_q.imag / self.omega),
                }
                result["energy_H"] = float(
                    np.vdot(y[1:], y[1:]).real + return_q.imag / self.omega
                )
                if result["energy_H"] < 0 or not np.isclose(
                    result["energy_H"], result["loop_H"], atol=1e-19, rtol=2e-6
                ):
                    raise ArithmeticError(
                        "Loop energy and reciprocal terminal impedance disagree"
                    )
                terms = result["energy_terms_H"]
                result["cancellation_ratio"] = (
                    sum(abs(v) for v in terms.values()) / result["energy_H"]
                    if result["energy_H"]
                    else None
                )
            if self.distance is not None:
                w = abs(flow) * self.sheet.length
                weight = w.sum()
                numerator = w @ (self.distance**2)
                for k, amps, q in zip(selected, barrel_currents, span_currents):
                    spoke = (
                        abs(np.diff(q, prepend=0, append=0)) * self.sites[k].spoke_mm
                    )
                    weight += sum(spoke)
                    numerator += spoke @ self.spoke_distances[k] ** 2
                    wvia = abs(amps) @ self.sites[k].heights
                    weight += wvia
                    numerator += wvia * self.site_distances[k] ** 2
                result["spread"] = (
                    float(np.sqrt(numerator / weight)) if weight else None
                )
        return result


def analyze(g, material, spec, legal_candidates):
    """Evaluate every remaining legal single addition at each greedy step.

    resistance uses DC ohms, potential uses abs(terminal voltage) in V at
    frequency_hz, and inductance uses the closed-loop Im(Z)/omega in H. Optional
    return_spacing_mm sets parallel-plane sheet inductance; via_return_radius_mm
    sets a coaxial return radius. signal_net supplies the projected geometry
    used to measure current-weighted RMS spread. A directed exciting_path_mm (xyz, sink toward source) additionally
    impresses a unit-current filament field per ampere of terminal current.
    Optional island_score(chosen_xy) returns an absolute score to minimize.
    """
    spec = dict(spec)
    obj = spec["objective"]
    units = {
        "resistance": "ohm",
        "inductance": "H",
        "potential": "V",
        "islands": spec.get("island_unit", "count"),
    }
    unit = units[obj]
    coords = sorted({tuple(map(float, xy)) for xy in legal_candidates})
    if any(len(x) != 2 or not np.isfinite(x).all() for x in coords):
        raise ValueError("Expected finite xy candidate coordinates")
    limit = spec["max_vias"]
    if isinstance(limit, bool) or int(limit) != limit or limit < 0:
        raise ValueError("max_vias must be a nonnegative integer")
    current = _number(spec.get("current_A", 1), "current_A", True)
    f = _number(spec.get("frequency_hz", 0), "frequency_hz")
    if obj == "inductance" and not f:
        raise ValueError("Positive frequency required for inductance")
    scorer = spec.get("island_score")
    if obj == "islands" and not callable(scorer):
        raise ValueError("islands requires island_score(chosen_xy)")
    electrical = obj != "islands" or "source" in spec or "sink" in spec
    spacing = _number(spec.get("min_spacing_mm", 0), "min_spacing_mm")
    networks, skipped, sites = {}, [], []
    if electrical:
        for k in ("mesh_mm", "drill_mm", "land_mm"):
            spec[k] = _number(spec[k], k, True)
        spec["plating_mm"] = _number(
            material.get("plating_mm", 0.025), "plating_mm", True
        )
        spec["return_spacing_mm"] = _number(
            spec.get("return_spacing_mm", 0), "return_spacing_mm"
        )
        if spec.get("via_return_radius_mm") is not None:
            spec["via_return_radius_mm"] = _number(
                spec["via_return_radius_mm"], "via_return_radius_mm", True
            )
        if spec["land_mm"] <= spec["drill_mm"]:
            raise ValueError("land_mm must exceed drill_mm")
        if "source" not in spec or "sink" not in spec:
            raise ValueError("Both source and sink are required")
        for layer in g["layers"]:
            _number(material["copper_mm"][layer], "foil thickness", True)
        _number(material.get("rho_ohm_m", 1.7241e-8), "resistivity", True)
        mesh = _sheet(g, material, spec)
        sites, skipped = _sites(mesh, coords, spec, spec["drill_mm"], spec["land_mm"])
        coords = [s.xy for s in sites]
        shape, path = _signal_shape(g, spec)
        distance = None
        if shape is not None:
            mid = (mesh.xy[mesh.a] + mesh.xy[mesh.b]) / 2
            distance = shapely.distance(shapely.points(mid), shape)
        for freq in sorted({0, f}):
            net = Network(mesh, sites, spec, freq, path, distance)
            if shape is not None:
                net.site_distances = [shape.distance(Point(site.xy)) for site in sites]
                net.spoke_distances = [
                    shapely.distance(
                        shapely.points((mesh.xy[list(site.nodes)] + site.xy) / 2), shape
                    )
                    for site in sites
                ]
            networks[freq] = net
    spacing = max(spacing, spec.get("drill_mm", 0))
    ranking = networks.get(0 if obj == "resistance" else f)
    missing_path = obj == "inductance" and not ranking.external
    if ranking and not missing_path and obj != "islands" and limit:
        ranking.prepare()

    def value(selected):
        if obj == "islands":
            score = float(scorer([list(coords[k]) for k in selected]))
            if not math.isfinite(score):
                raise ValueError("Island score must be finite")
            return score
        if missing_path:
            return None
        info = ranking.evaluate(selected)
        if info is None:
            return None
        if obj == "resistance":
            return float(info["z"].real)
        if obj == "inductance":
            return info["loop_H"]
        if ranking.external and info["loop_H"] is None:
            return None
        return float(abs(info["v"]) * current)

    def describe(selected, label=None):
        meta, maps = {}, []
        for freq, net in networks.items():
            info = net.evaluate(selected, state=True)
            prefix = "ac" if freq else "dc"
            meta[prefix + "_connected"] = info is not None
            if info is None:
                continue
            meta[prefix + "_potential_V"] = float(abs(info["v"]) * current)
            if freq:
                meta.update(
                    loop_inductance_H=info["loop_H"],
                    loop_status=info["loop_status"],
                    forward_self_H=net.forward_h,
                    magnetic_basis_rank=len(net.modes.forward) if net.external else 0,
                    loop_energy_H=info.get("energy_H"),
                    loop_energy_terms_H=info.get("energy_terms_H"),
                    energy_cancellation_ratio=info.get("cancellation_ratio"),
                    passivity=info.get("loop_passivity"),
                )
                if net.external and info["loop_H"] is None:
                    meta["ac_potential_V"] = None
                    continue
                meta["ac_resistance_ohm"] = float(info["z"].real)
                meta["port_inductance_H"] = float(info["z"].imag / (math.tau * freq))
            else:
                meta["resistance_ohm"] = float(info["z"].real)
            if "spread" in info:
                meta[prefix + "_rms_spread_mm"] = info["spread"]
            meta[prefix + "_added_span_current_A"] = (
                current * abs(info["barrels"])
            ).tolist()
            if label:
                title = f"{label} {freq:g} Hz"
                pitch = spec.get("map_pitch_mm", max(1.0, mesh.h))
                active = info["active"]
                maps += mapset(
                    title + " potential",
                    "V",
                    mesh.levels[active],
                    mesh.xy[active],
                    current * abs(info["volts"][active]),
                    pitch,
                )
                edges = np.ones_like(mesh.a, bool) if net.external else active[mesh.a]
                density = current * abs(info["flow"]) / (mesh.width * mesh.thick)
                maps += mapset(
                    title + " current density",
                    "A/mm2",
                    mesh.levels[mesh.a[edges]],
                    (mesh.xy[mesh.a[edges]] + mesh.xy[mesh.b[edges]]) / 2,
                    density[edges],
                    pitch,
                )
                for k, amps in zip(
                    selected,
                    np.split(
                        current * abs(info["barrels"]),
                        np.cumsum([len(sites[k].heights) for k in selected])[:-1],
                    ),
                ):
                    for layer, amp in zip(sites[k].layers, amps):
                        maps.append(
                            {
                                "name": title + " added barrel current",
                                "layer": layer,
                                "unit": "A",
                                "points": [[*coords[k], float(amp)]],
                            }
                        )
        return meta, maps

    chosen, rows, stages = [], [], []
    before = value(chosen)
    baseline = before
    previous, maps = describe(chosen, "Before")
    base_metrics = previous
    remaining = list(range(len(coords)))
    for step in range(1, int(limit) + 1):
        if not remaining:
            break
        scores = [value([*chosen, k]) for k in remaining]
        candidates = [
            {
                "xy": list(coords[k]),
                "after": v,
                "improvement": None if before is None or v is None else before - v,
            }
            for k, v in zip(remaining, scores)
        ]
        stages.append(
            {"step": step, "before": before, "unit": unit, "candidates": candidates}
        )
        layers = spec.get("layers", g.get("layers", []))
        gain_points = [
            [*item["xy"], item["improvement"]]
            for item in candidates
            if item["improvement"] is not None
        ]
        if gain_points and layers:
            maps.append(
                {
                    "name": f"Step {step} candidate improvement",
                    "unit": unit,
                    "layer": layers[0],
                    "points": gain_points,
                }
            )
        finite = [(v, k) for k, v in zip(remaining, scores) if v is not None]
        if not finite:
            break
        after, k = min(finite)
        if before is not None and after >= before - 1e-12 * max(
            abs(before), abs(after)
        ):
            break
        for net in networks.values():
            net.accept(k)
        chosen.append(k)
        now, _ = describe(chosen)
        rows.append(
            {
                "step": step,
                "xy": list(coords[k]),
                "before": before,
                "after": after,
                "improvement": None if before is None else before - after,
                "unit": unit,
                "metrics_before": previous,
                "metrics_after": now,
            }
        )
        previous, before = now, after
        remaining = [
            i
            for i in remaining
            if i != k and math.dist(coords[k], coords[i]) >= spacing
        ]
    if chosen:
        _, final_maps = describe(chosen, "After")
        maps.extend(final_maps)
    return {
        "name": spec.get("name", "Stitch placement"),
        "objective": obj,
        "unit": unit,
        "baseline": baseline,
        "final": before,
        "frequency_hz": f,
        "current_A": current,
        "rows": rows,
        "stages": stages,
        "baseline_metrics": base_metrics,
        "final_metrics": previous,
        "skipped": skipped,
        "maps": maps,
        "model": {
            "selection": "Best single addition per stage. Joint multi-via optimum and improvements requiring neutral first steps are not guaranteed.",
            "contacts": "Finite radial spokes to cells at each contacted rim, fixed foil between holes. Check rankings on a finer mesh and after repour.",
            "terminals": "Named pads are equipotential on every resolved pad layer. A single-layer disk selects a particular injection surface.",
            "forward": "Package closures are straight centroid-to-endpoint links. The forward path has a specified effective radius; its loss and device impedance are omitted.",
            "mesh_mm": spec.get("mesh_mm"),
            "bounds_mm": spec.get("bounds_mm"),
            "wire_radius_mm": spec.get("wire_radius_mm", 0.08),
            "filament_step_mm": spec.get("filament_step_mm", 0.4),
            "return_spacing_mm": spec.get("return_spacing_mm", 0),
            "via_return_radius_mm": spec.get("via_return_radius_mm"),
            "ac": "Symmetric slab and optional local plane/coaxial leakage. Closed-loop magnetic energy projects foil and barrel currents onto the forward and terminal-chord filament fields; return_modes_mm adds probe paths. Orthogonal external fields are unresolved. Check basis and mesh sensitivity. No resonance or full-wave extraction.",
            "drive": "Explicit directed filament and two terminals"
            if spec.get("exciting_path_mm") is not None
            else "Two terminals; signal tracks, if given, are only a spatial spread reference.",
            "uncertainty": "Current amplitude, spectrum, return closure and component parasitics must be supplied or measured for a physical waveform prediction.",
        },
    }
