"""Quasi-static capacitance matrices of one or two rectangular conductors (SI outputs)."""

from functools import lru_cache
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1 / (EPS0 * C0**2)


def _axis(breaks, pitch):
    points = sorted(set(round(float(v), 10) for v in breaks))
    vertices = []
    for left, right in zip(points, points[1:]):
        half = (right - left) / 2
        local = [0.0]
        while local[-1] < half:
            distance = local[-1]
            step = min(pitch * 3, pitch / 8 + distance * 0.16)
            if half - distance < step * 1.45:
                local.append(half)
                break
            local.append(distance + step)
        offsets = np.array(local)
        vertices.extend(left + offsets[:-1])
        vertices.extend(right - offsets[:0:-1])
    return np.r_[vertices, points[-1]]


@lru_cache(maxsize=192)
def section(
    widths,
    gap,
    height,
    upper,
    thickness,
    er,
    loss_tangent,
    mask,
    mask_er,
    mask_tangent,
    cells=12,
    margin=7,
):
    """Ground beneath at -height; optional upper ground at thickness+upper.

    The vertices follow each dielectric and conductor edge. Outer side walls and
    the distant top of an open section are grounded. Repeat with larger margin
    and a higher cell count to check truncation and grid error.
    """
    widths = tuple(widths)
    if (
        not 1 <= len(widths) <= 2
        or min(widths) <= 0
        or height <= 0
        or thickness < 0
        or upper < 0
        or er < 1
        or mask_er < 1
        or min(mask, loss_tangent, mask_tangent) < 0
        or cells < 4
        or margin < 2
        or (len(widths) == 2 and gap <= 0)
    ):
        raise ValueError("Invalid cross-section dimensions or material")
    edges = []
    pos = 0.0
    for w in widths:
        edges.append((pos, pos + w))
        pos += w + gap
    span = min(height, upper or height)
    pitch = min(span, min(widths)) / cells
    extent = max(height, upper) * margin
    top = thickness + (upper or (extent + mask))
    x = _axis(
        [-extent, edges[-1][1] + extent, *(v for edge in edges for v in edge)], pitch
    )
    y = _axis(
        [-height, 0, thickness, top]
        + ([thickness + mask] if mask and not upper else []),
        pitch,
    )
    nx, ny = len(x), len(y)
    if nx * ny > 350000:
        raise ValueError(
            "Cross section too large; choose explicit mesh resolution or section extent"
        )
    dx, dy = np.diff(x), np.diff(y)
    nodes = np.arange(nx * ny).reshape(ny, nx)
    ll = nodes[:-1, :-1]
    a = np.concatenate([v.ravel() for v in (ll, ll + nx, ll, ll + 1)])
    b = np.concatenate([v.ravel() for v in (ll + 1, ll + nx + 1, ll + nx, ll + nx + 1)])
    # Sum the four half-cell tangential energies; discontinuities coincide with
    # cell edges so each half has the correct dielectric.
    horizontal = dy[:, None] / (2 * dx)
    vertical = dx / (2 * dy[:, None])
    weights = np.concatenate(
        [v.ravel() for v in (horizontal, horizontal, vertical, vertical)]
    )
    mid_y = (y[:-1] + y[1:]) / 2
    dielectric = (mid_y < 0) | bool(upper)
    coating = (~dielectric) & (mid_y < thickness + mask) & (mask > 0)

    def layer_values(base, coat, air):
        data = np.where(dielectric, base, np.where(coating, coat, air))
        return np.tile(np.repeat(data, nx - 1), 4)

    perm = layer_values(er, mask_er, 1)
    tangent = layer_values(loss_tangent, mask_tangent, 0)
    label = np.full((ny, nx), -2)
    label[[0, -1], :] = -1
    label[:, [0, -1]] = -1
    for i, (left, right) in enumerate(edges):
        label[
            np.ix_(
                (y >= -1e-9) & (y <= thickness + 1e-9),
                (x >= left - 1e-9) & (x <= right + 1e-9),
            )
        ] = i
    fixed = label.ravel() != -2

    def potentials(relative):
        g = weights * relative
        stiffness = sparse.coo_matrix(
            (np.r_[g, g, -g, -g], (np.r_[a, b, a, b], np.r_[a, b, b, a])),
            shape=(nx * ny, nx * ny),
        ).tocsc()
        volt = np.zeros((nx * ny, len(widths)))
        for i in range(len(widths)):
            volt[label.ravel() == i, i] = 1
        volt[~fixed] = splu(stiffness[~fixed][:, ~fixed]).solve(
            -(stiffness[~fixed] @ volt)
        )
        drop = volt[a] - volt[b]
        return EPS0 * (drop.T @ (g[:, None] * drop)), drop

    cap0, _ = potentials(1)
    cap, field = potentials(perm)
    loss = EPS0 * (field.T @ ((weights * perm * tangent)[:, None] * field))
    ind = np.linalg.inv(cap0) / C0**2
    return cap, ind, loss


def modal(cap, ind):
    """Driving-point values for a uniform section with modal symmetry."""
    result = {}
    modes = (
        {"single": (np.array([1.0]), np.array([1.0]))}
        if len(cap) == 1
        else {
            "diff": (np.array([0.5, -0.5]), np.array([1.0, -1.0])),
            "common": (np.array([1.0, 1.0]), np.array([0.5, 0.5])),
        }
    )
    for name, (volts, amps) in modes.items():
        c = float(volts @ cap @ volts)
        l = float(amps @ ind @ amps)
        result[name] = {"z_ohm": math.sqrt(l / c), "velocity_m_s": 1 / math.sqrt(l * c)}
    return result
