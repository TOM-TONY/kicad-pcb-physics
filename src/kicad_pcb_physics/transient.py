"""Local transient replay with numerical substitutions from extracted copper loops."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(spec, directory, data, geo, simulator):
    template = (directory / spec["template"]).read_text()
    numbers = {}
    for key, ref in spec.get("parameters", {}).items():
        if isinstance(ref, (float, int)):
            numbers[key] = ref
        else:
            port = next(case for case in data if case["name"] == ref["case"])
            values = port["loops"][ref["branch"]]
            numbers[key] = float(values[ref["quantity"]]) * ref.get(
                "scale", 1
            ) + ref.get("offset", 0)
        template = template.replace("@@" + key + "@@", format(numbers[key], ".16g"))
    if "@@" in template:
        raise ValueError("Unresolved numerical placeholder")
    kind = spec.get("backend", simulator.get("backend", "ngspice"))
    if kind not in ("ngspice", "ltspice"):
        raise ValueError("Unsupported transient simulator")
    binary = spec.get("executable") or (
        simulator.get("executable")
        if simulator.get("backend", "ngspice") == kind
        else None
    )
    binary = binary or os.environ.get(kind.upper() + "_BIN") or shutil.which(kind)
    if not binary:
        raise FileNotFoundError("Set simulator executable for " + kind)
    with tempfile.TemporaryDirectory(prefix="pcb-replay-") as tmp:
        folder = Path(tmp)
        for name in spec.get("includes", []):
            source = (directory / name).resolve()
            shutil.copyfile(source, folder / source.name)
        deck = folder / "response.cir"
        deck.write_text(template)
        proc = subprocess.run(
            [binary, "-b", deck.name],
            cwd=folder,
            capture_output=True,
            timeout=spec.get("timeout_s", 180),
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-1500:])
        if kind == "ngspice":
            log = proc.stdout.decode(errors="replace")
        else:
            text = deck.with_suffix(".log").read_bytes()
            log = text.decode(
                "utf-16" if text.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8",
                errors="replace",
            ).replace("\x00", "")
    values = {}
    maps = []
    for metric in spec["measurements"]:
        label = metric["name"]
        token = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        pattern = (
            r"(?im)^\s*" + re.escape(label) + r"(?::[^\n]*?)?\s*=\s*(" + token + r")"
        )
        match = re.search(pattern, log)
        if not match:
            raise ValueError("No valid transient measurement: " + label)
        value = float(match.group(1))
        values[label] = value
        for ref in metric.get("parts", []):
            part = geo["parts"][ref]
            maps.append(
                {
                    "name": spec["name"] + " / " + label,
                    "unit": metric["unit"],
                    "layer": part["layer"],
                    "points": [[*part["xy"], value]],
                }
            )
    return {
        "name": spec["name"],
        "backend": kind,
        "parameters": numbers,
        "measurements": values,
        "deck_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "maps": maps,
    }
