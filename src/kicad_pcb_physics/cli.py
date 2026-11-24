"""Analyze a filled board with local, user-supplied material and stimulus data."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from . import fields, heat, pdn, spice, studies, transient, stitching, transmission
from .mesh import polygon
from contextlib import contextmanager


def interpreter(config):
    explicit = config.get("kicad_python") or os.environ.get("KICAD_PYTHON")
    if explicit:
        return explicit
    candidates = [sys.executable, shutil.which("python3"), "/usr/bin/python3"]
    if sys.platform == "darwin":
        candidates += list(
            Path("/Applications").glob(
                "KiCad*/**/Python.framework/Versions/*/bin/python3"
            )
        )
    for candidate in dict.fromkeys(c for c in candidates if c):
        if (
            subprocess.run(
                [str(candidate), "-B", "-c", "import pcbnew"], capture_output=True
            ).returncode
            == 0
        ):
            return str(candidate)
    raise FileNotFoundError("Set KICAD_PYTHON to an interpreter that imports pcbnew")


def load(board, config, folder):
    cache = folder / "geometry.json"
    bridge = Path(__file__).with_name("bridge.py")
    key = hashlib.sha256(board.read_bytes() + bridge.read_bytes()).hexdigest()
    if cache.is_file():
        g = json.loads(cache.read_text())
        if g.get("cache_key") == key:
            return g
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geometry.json"
        subprocess.run([interpreter(config), "-B", bridge, board, path], check=True)
        g = json.loads(path.read_text())
    g["cache_key"] = key
    save(cache, g)
    return g


def save(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


@contextmanager
def report_lock(folder):
    # The OS releases the lock if a writer exits during installation.
    with (folder / ".report.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if not handle.tell():
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def run(board, config, output, *, proposal=None, stitch_case=None):
    board, config, folder = map(Path, (board, config, output))
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.is_file() and any(path.samefile(src) for src in (board, config)):
            raise ValueError("Keep board and conditions outside the output directory")
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        staged = Path(tmp) / "report"
        staged.mkdir()
        old_cache = folder / "geometry.json"
        if old_cache.is_file():
            shutil.copyfile(old_cache, staged / old_cache.name)
        file = Path(tmp) / "board.kicad_pcb"
        file.write_bytes(board.read_bytes())
        data = config.read_bytes()
        settings = json.loads(data)
        board_hash = hashlib.sha256(file.read_bytes()).hexdigest()
        config_hash = hashlib.sha256(data).hexdigest()
        results, exporter = analyze_snapshot(file, settings, config, staged, board_hash)
        if (
            hashlib.sha256(board.read_bytes()).hexdigest() != board_hash
            or hashlib.sha256(config.read_bytes()).hexdigest() != config_hash
        ):
            raise ValueError(
                "Inputs changed during analysis; repeat with stable inputs"
            )
        save(
            staged / "provenance.json",
            {
                "board_sha256": board_hash,
                "configuration_sha256": config_hash,
                "exporter": exporter,
                "output_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(staged.iterdir())
                    if p.is_file()
                },
            },
        )
        if proposal:
            cases = [
                (s, r)
                for s, r in zip(settings.get("stitching", []), results["stitching"])
                if s["name"] == stitch_case
            ]
            if len(cases) != 1 or not cases[0][1]["rows"]:
                raise ValueError("Select one completed case with recommended vias")
            spec, chosen = cases[0]
            path = Path(tmp) / "selection.json"
            save(
                path,
                {k: spec[k] for k in ("net", "land_mm", "drill_mm")}
                | {"coordinates": [r["xy"] for r in chosen["rows"]]},
            )
            subprocess.run(
                [
                    interpreter(settings),
                    "-B",
                    Path(__file__).with_name("via_copy.py"),
                    file,
                    proposal,
                    path,
                ],
                check=True,
            )
        with report_lock(folder):
            # Publish the completion record last. An interrupted installation can
            # never certify mixed files with the previous run's input hashes.
            marker = folder / "provenance.json"
            marker.unlink(missing_ok=True)
            for artifact in staged.iterdir():
                if artifact.name != marker.name:
                    artifact.replace(folder / artifact.name)
            (staged / marker.name).replace(marker)
        return results


def analyze_snapshot(file, settings, config, folder, original_sha):
    geo = load(file, settings, folder)
    mat = settings["material"]
    if set(mat["depth_mm"]) != set(geo["layers"]) or set(mat["copper_mm"]) != set(
        geo["layers"]
    ):
        raise ValueError("Material and board layers differ")
    mat["depth_mm"] = {layer: mat["depth_mm"][layer] for layer in geo["layers"]}
    results = {
        "electrical": [],
        "thermal": [],
        "pdn": [],
        "receivers": [],
        "returns": [],
        "radiation": [],
        "exposure": [],
        "transients": [],
    }
    for case in settings.get("electrical", []):
        results["electrical"].append(studies.electrical(geo, mat, case))
    for name, cfg in settings.get("thermal", {}).items():
        electrical_losses = [
            e for e in results["electrical"] if e["name"] in cfg.get("dc_cases", [])
        ]
        if "loss_cells_file" in cfg:
            auxiliary = json.loads(
                (Path(config).resolve().parent / cfg["loss_cells_file"]).read_text()
            )
            if auxiliary["board_sha256"] != original_sha:
                raise ValueError("Loss field and input board differ")
            electrical_losses.append({"heat_cells": auxiliary["heat_cells"]})
        data = heat.analyze(geo, cfg, electrical_losses)
        for m in data["maps"]:
            m["name"] = name + " / " + m["name"]
        results["thermal"].append({"name": name, **data})
    for spec in settings.get("pdn", []):
        data = pdn.analyze(geo, mat, spec)
        if settings.get("simulator", {}).get("enabled", True):
            data["spice"] = spice.check(
                data, spec["feed"], settings.get("simulator", {})
            )
            if data["spice"]["max_relative_error"] > settings.get(
                "spice_relative_tolerance", 1e-4
            ):
                raise ArithmeticError("Circuit comparison outside requested tolerance")
        results["pdn"].append(data)
    for spec in settings.get("receivers", []):
        results["receivers"].append(studies.reception(geo, mat, spec))
    for spec in settings.get("transients", []):
        results["transients"].append(
            transient.run(
                spec,
                Path(config).resolve().parent,
                results["pdn"],
                geo,
                settings.get("simulator", {}),
            )
        )
    for spec in settings.get("exposure", []):
        results["exposure"].append(fields.exposure(geo, mat, spec))
    for spec in settings.get("returns", []):
        gaps = fields.return_gaps(
            geo,
            spec["signal"],
            spec["reference"],
            spec["layer"],
            spec["reference_layer"],
        )
        results["returns"].append(
            {
                "name": spec["name"],
                "gaps": gaps,
                "maps": [
                    {
                        "name": spec["name"],
                        "unit": "mm",
                        "layer": spec["layer"],
                        "points": [[*g["centre_mm"], g["missing_mm"]] for g in gaps],
                    }
                ],
            }
        )
    for spec in settings.get("radiation", []):
        if "pdn_case" in spec:
            branch = next(s for s in results["pdn"] if s["name"] == spec["pdn_case"])[
                "loops"
            ][spec["branch"]]
            curve = branch["path_mm"]
        else:
            curve = spec["closed_path_mm"]
        estimates = [
            {
                "frequency_Hz": f,
                **fields.dipole_limit(
                    np.array(curve) * 0.001, spec["current_A"], f, spec["distance_m"]
                ),
            }
            for f in spec["frequency_Hz"]
        ]
        maps = []
        for row in estimates:
            if row["applicable"]:
                for layer in geo["layers"]:
                    pts = [
                        [*p[:2], row["maximum_far_E_V_m"]]
                        for p in curve
                        if abs(p[2] - mat["depth_mm"][layer]) < 1e-6
                    ]
                    if pts:
                        maps.append(
                            {
                                "name": spec["name"]
                                + " "
                                + str(row["frequency_Hz"])
                                + " Hz source",
                                "unit": "V/m",
                                "layer": layer,
                                "points": pts,
                            }
                        )
        results["radiation"].append(
            {"name": spec["name"], "rows": estimates, "maps": maps}
        )
    if "transmission" in settings:
        lines, matches = transmission.studies(
            geo, mat, settings["transmission"], folder
        )
        results["transmission"] = lines
        results["matching"] = matches
    stitching_cases = settings.get("stitching", [])
    results["stitching"] = [
        stitching.analyze(geo, mat, case) for case in stitching_cases
    ]
    if stitching_cases:
        with (folder / "via-proposals.csv").open("w", newline="") as fd:
            writer = csv.writer(fd)
            writer.writerow(
                [
                    "case",
                    "step",
                    "x_mm",
                    "y_mm",
                    "before",
                    "after",
                    "improvement",
                    "unit",
                ]
            )
            for result in results["stitching"]:
                for row in result["rows"]:
                    writer.writerow(
                        [
                            result["name"],
                            row["step"],
                            *row["xy"],
                            row["before"],
                            row["after"],
                            row["improvement"],
                            row["unit"],
                        ]
                    )
        with (folder / "via-candidates.csv").open("w", newline="") as fd:
            writer = csv.writer(fd)
            writer.writerow(
                [
                    "case",
                    "step",
                    "rank",
                    "x_mm",
                    "y_mm",
                    "before",
                    "after",
                    "improvement",
                    "unit",
                ]
            )
            for result in results["stitching"]:
                for stage in result["stages"]:
                    scores = sorted(
                        stage["candidates"],
                        key=lambda r: (r["after"] is None, r["after"], r["xy"]),
                    )
                    for rank, row in enumerate(scores, 1):
                        writer.writerow(
                            [
                                result["name"],
                                stage["step"],
                                rank,
                                *row["xy"],
                                stage["before"],
                                row["after"],
                                row["improvement"],
                                stage["unit"],
                            ]
                        )
    save(folder / "results.json", results)
    with (folder / "impedance.csv").open("w", newline="") as fd:
        w = csv.writer(fd)
        w.writerow(["case", "frequency_Hz", "magnitude_ohm", "phase_deg"])
        for p in results["pdn"]:
            for row in zip(p["frequency_Hz"], p["z_abs_ohm"], p["phase_deg"]):
                w.writerow([p["name"], *row])
    maps = [
        m
        for family in results.values()
        for case in family
        for m in case.get("maps", [])
        if m["points"]
    ]
    # Simplify display copper only; the solver uses the full original contours.
    metal = {l: [] for l in geo["layers"]}
    for n in geo["nets"].values():
        for l, records in n["copper"].items():
            for shape in records:
                g = polygon([shape]).simplify(0.03, preserve_topology=True)
                for poly in list(g.geoms) if g.geom_type == "MultiPolygon" else [g]:
                    if poly.geom_type == "Polygon":
                        metal[l].append(
                            [
                                list(poly.exterior.coords),
                                *[list(r.coords) for r in poly.interiors],
                            ]
                        )
    display_maps = []
    for m in maps:
        pts = m["points"]
        if len(pts) > 18000 and m["unit"] == "degC":
            arr = np.asarray(pts)
            pitch = settings.get("display_pitch_mm", 0.4)
            bins, idx = np.unique(
                np.floor(arr[:, :2] / pitch).astype(int), axis=0, return_inverse=True
            )
            maxima = np.full(len(bins), -np.inf)
            np.maximum.at(maxima, idx, arr[:, 2])
            display_maps.append(
                {
                    **m,
                    "points": np.column_stack(((bins + 0.5) * pitch, maxima)).tolist(),
                }
            )
        else:
            display_maps.append(m)
    view = {
        "layers": geo["layers"],
        "outline": geo["outline"],
        "metal": metal,
        "maps": display_maps,
        "lines": [
            {
                k: r[k]
                for k in (
                    "name",
                    "files",
                    "length_mm",
                    "time_ps",
                    "skew_ps",
                    "propagation_model",
                )
            }
            for r in results.get("transmission", [])
        ],
        "stitch_count": len(results["stitching"]),
        "pdn": [
            {
                k: p[k]
                for k in ("name", "frequency_Hz", "z_abs_ohm", "ripple_pp_V")
                if k in p
            }
            for p in results["pdn"]
        ],
    }
    (folder / "data.js").write_text(
        "window.boardView="
        + json.dumps(view, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        + ";\n"
    )
    shutil.copyfile(Path(__file__).with_name("viewer.html"), folder / "index.html")
    return results, geo["exporter_version"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path)
    p.add_argument("settings", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--write-vias", type=Path, help="Write selected vias to a new board copy"
    )
    p.add_argument("--stitch-case", help="Unique stitching case to export")
    args = p.parse_args()
    if bool(args.write_vias) != bool(args.stitch_case):
        p.error("Specify --write-vias and --stitch-case together")
    if args.write_vias and (
        args.write_vias.exists() or args.write_vias.resolve() == args.board.resolve()
    ):
        p.error("The proposal path must be new")
    run(
        args.board,
        args.settings,
        args.output,
        proposal=args.write_vias,
        stitch_case=args.stitch_case,
    )
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
