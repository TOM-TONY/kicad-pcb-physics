"""Frequency-domain multiconductor line cascade with equal real port references."""

import math
import numpy as np
from scipy.linalg import block_diag, expm

from .cross_section import MU0
from .mesh import slab


def rl(section, frequency, rho, roughness):
    w = np.array(section["widths"]) * 1e-3
    metal = np.array(section["thickness"]) * 1e-3
    # Slab current diffusion, with a finite reference-plane return width.
    # This omits proximity crowding at trace edges and mutual conductor loss.
    skin = math.sqrt(rho / (math.pi * frequency * MU0))
    rough = 1 + 2 / math.pi * math.atan(1.4 * (roughness * 1e-6 / skin) ** 2)
    surface = np.array([slab(frequency, t, rho) for t in metal])
    surface = surface.real * rough + 1j * surface.imag
    return surface * (1 / w + 1 / (w + 2 * np.array(section["heights"]) * 1e-3))


def response(
    sections, freq, zref, rho, roughness, conductor_loss=True, dielectric_loss=True
):
    n = len(sections[0]["widths"])
    eye = np.eye(n)
    zero = np.zeros_like(eye)
    transfer = np.eye(2 * n, dtype=complex)
    omega = 2 * math.pi * freq
    absorbed = np.zeros(n)
    for seg in sections:
        d = np.diag(np.sqrt(np.array(seg["lengths"]) * 1e-3))
        internal = rl(seg, freq, rho, roughness)
        if not conductor_loss:
            internal = 1j * internal.imag
        z = np.diag(internal) + 1j * omega * seg["L"]
        y = omega * ((seg["loss"] if dielectric_loss else 0) + 1j * seg["C"])
        transfer = transfer @ expm(np.block([[zero, d @ z @ d], [d @ y @ d, zero]]))
        beta = np.sqrt(np.diag(z) * np.diag(y))
        absorbed += 8.685889638 * beta.real * np.diag(d) ** 2
        if any(seg["stubs"]):
            # Lossless open-barrel stubs are a quasi-TEM approximation.
            stub_y = []
            for lengths, velocity, imp in zip(
                seg["stubs"], seg["speeds"], seg["stub_z"]
            ):
                stub_y.append(
                    sum(
                        1j * np.tan(omega * mm * 1e-3 / velocity) / imp
                        for mm in lengths
                        if mm > 0
                    )
                )
            transfer = transfer @ np.block([[eye, zero], [np.diag(stub_y), eye]])
    w = np.block(
        [
            [eye * math.sqrt(zref), eye * math.sqrt(zref)],
            [eye / math.sqrt(zref), -eye / math.sqrt(zref)],
        ]
    )
    m = np.linalg.solve(w, transfer @ w)
    forward = np.linalg.solve(m[:n, :n], eye)
    reverse = -forward @ m[:n, n:]
    s = np.block(
        [[m[n:, :n] @ forward, m[n:, n:] + m[n:, :n] @ reverse], [forward, reverse]]
    )
    if n == 2:
        pair = np.array([[1, -1], [1, 1]]) / math.sqrt(2)
        t = block_diag(pair, pair)
        mixed = t @ s @ t.T
    else:
        mixed = None
    return s, mixed, absorbed


def touchstone(path, freqs, matrices, zref, mixed=False):
    n = len(matrices[0])
    with path.open("w", encoding="ascii") as f:
        f.write(f"[Version] 2.0\n# Hz S RI R {zref:g}\n[Number of Ports] {n}\n")
        if n == 2:
            f.write("[Two-Port Data Order] 12_21\n")
        f.write(f"[Number of Frequencies] {len(freqs)}\n[Matrix Format] Full\n")
        if mixed:
            f.write("[Mixed-Mode Order] D1,2 C1,2 D3,4 C3,4\n")
        f.write("[Network Data]\n")
        for freq, s in zip(freqs, matrices):
            for index, row in enumerate(s):
                prefix = f"{freq:.12g} " if index == 0 else "  "
                f.write(
                    prefix
                    + " ".join(f"{v.real:.12g} {v.imag:.12g}" for v in row)
                    + "\n"
                )
        f.write("[End]\n")
