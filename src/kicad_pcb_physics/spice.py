"""Delegate a passive multi-branch one-port to a separately installed simulator."""

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np


def circuit(data, feed):
    rows = [
        "Port admittance cross-check",
        "Itest 0 input AC 1",
        f"Rfeed input wire {feed['resistance_ohm']}",
        f"Lfeed wire 0 {feed['inductance_H']}",
    ]
    L = np.array(data["inductance_matrix_H"])
    for i, (r, c) in enumerate(zip(data["branch_R_ohm"], data["effective_F"])):
        rows.append(f"Vprobe{i} input sense{i} 0")
        node = f"r{i}"
        rows.append(f"R{i} sense{i} {node} {r:.14g}")
        for j, row in enumerate(data["resistance_matrix_ohm"][i]):
            if i == j or not row:
                continue
            other = f"h{i}_{j}"
            rows.append(f"H{i}_{j} {node} {other} Vprobe{j} {row:.14g}")
            node = other
        rows += [f"L{i} {node} c{i} {L[i, i]:.14g}", f"C{i} c{i} 0 {c:.14g}"]
        for j in range(i):
            coupling = L[i, j] / math.sqrt(L[i, i] * L[j, j])
            if abs(coupling) > 1 + 1e-6:
                raise ValueError("Invalid coupling")
            if coupling:
                rows.append(f"K{i}_{j} L{i} L{j} {coupling:.14g}")
    return rows


def check(data, feed, cfg):
    kind = cfg.get("backend", "ngspice")
    if kind not in ("ngspice", "ltspice"):
        raise ValueError("Unsupported simulator")
    exe = (
        cfg.get("executable")
        or os.environ.get(kind.upper() + "_BIN")
        or shutil.which(kind)
    )
    if not exe:
        raise FileNotFoundError(f"Set {kind.upper()}_BIN or simulator.executable")
    f = np.asarray(data["frequency_Hz"])
    with tempfile.TemporaryDirectory(prefix="pcb-circuit-") as temp:
        folder = Path(temp)
        base = circuit(data, feed)
        deck = folder / "port.cir"
        if kind == "ngspice":
            commands = [".control", "set appendwrite", "set numdgt=15"]
            for freq in f:
                commands.extend(
                    [f"ac lin 1 {freq:.15g} {freq:.15g}", "wrdata samples.txt v(input)"]
                )
            commands.extend(["quit", ".endc", ".end"])
            deck.write_text("\n".join(base + commands) + "\n")
            run = subprocess.run(
                [exe, "-b", deck.name], cwd=folder, capture_output=True, timeout=180
            )
            output = folder / "samples.txt"
            if run.returncode or not output.exists():
                raise RuntimeError(
                    run.stdout.decode(errors="replace")[-1200:]
                    + run.stderr.decode(errors="replace")[-500:]
                )
            vals = np.loadtxt(output, ndmin=2)
            if not np.allclose(vals[:, 0], f):
                raise ValueError("AC sweep frequencies differ")
            mag = abs(vals[:, 1] + 1j * vals[:, 2])
        else:
            # Independent single-frequency solves preserve arbitrary requested frequencies.
            base = [line + " Rser=0" if line.startswith("L") else line for line in base]
            mag = []
            for freq in f:
                deck.write_text(
                    "\n".join(
                        base
                        + [
                            ".options meascplxfmt=cartesian numdgt=15",
                            f".ac lin 1 {freq:.15g} {freq:.15g}",
                            f".meas ac voltage FIND mag(V(input)) at={freq:.15g}",
                            ".end",
                        ]
                    )
                    + "\n"
                )
                subprocess.run(
                    [exe, "-b", deck.name],
                    cwd=folder,
                    capture_output=True,
                    check=True,
                    timeout=180,
                )
                blob = deck.with_suffix(".log").read_bytes()
                log = blob.decode(
                    "utf-16"
                    if blob.startswith((b"\xff\xfe", b"\xfe\xff"))
                    else "utf-8",
                    errors="replace",
                ).replace("\x00", "")
                match = re.search(
                    r"(?im)^voltage:.*?=\s*\(?([-+.0-9eE]+)(?:\s*,\s*[-+.0-9eE]+)?\)?(?:\s|$)",
                    log,
                )
                if not match:
                    raise RuntimeError("No complex measurement in simulator log")
                mag.append(float(match[1]))
            mag = np.array(mag)
    expected = np.array(data["z_abs_ohm"])
    return {
        "backend": kind,
        "frequency_Hz": f.tolist(),
        "z_ohm": mag.tolist(),
        "max_relative_error": float(max(abs(mag / expected - 1))),
    }
