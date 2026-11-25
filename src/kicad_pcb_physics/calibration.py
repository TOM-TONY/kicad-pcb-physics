"""Calibrate a discrete joint parameter sweep against bounded observations."""

import argparse
import json
import math
from pathlib import Path


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return value


def named(rows):
    result = {}
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in result or not row["unit"]:
            raise ValueError("Each quantity requires a unique id and a unit")
        result[key] = row
    return result


def calibrate(data):
    observations = named(data["observations"])
    targets = named(data.get("targets", []))
    if observations.keys() & targets.keys():
        raise ValueError("Target and observation ids overlap")
    for row in observations.values():
        if number(row["low"]) > number(row["high"]) or number(row["scale"]) <= 0:
            raise ValueError("Invalid observation bounds or scale")
        if row["role"] not in ("fit", "check"):
            raise ValueError("Expected a fit or check role")
    fit = [key for key, row in observations.items() if row["role"] == "fit"]
    if not fit or not data["runs"]:
        raise ValueError("At least one fit observation and one model run are required")
    for target in targets.values():
        for key in target["error_from"]:
            if key not in observations or observations[key]["unit"] != target["unit"]:
                raise ValueError("Transfer residuals require observations in matching units")

    records = []
    runs = data["runs"]
    parameters = runs[0]["parameters"].keys()
    if not parameters:
        raise ValueError("A run must identify its parameters")
    for i, run in enumerate(runs):
        if run["parameters"].keys() != parameters:
            raise ValueError("Inconsistent parameter dimensions")
        for value in run["parameters"].values():
            number(value)
        predictions = run["predictions"]
        for key in observations.keys() | targets.keys():
            number(predictions[key])
        # Signed data-minus-model interval. Zero is compatible with the data.
        residual = {key: [row["low"] - predictions[key], row["high"] - predictions[key]]
                    for key, row in observations.items()}
        miss = {key: max(0, lo, -hi) / observations[key]["scale"]
                for key, (lo, hi) in residual.items()}
        records.append({"index": i, "max_fit_miss": max(miss[key] for key in fit),
                        "residual": residual, "normalized_miss": miss})

    best = min(record["max_fit_miss"] for record in records)
    accepted = [r for r in records if r["max_fit_miss"] == 0]
    result = {
        "run_count": len(records), "accepted_indices": [r["index"] for r in accepted],
        "closest_indices": [r["index"] for r in records if r["max_fit_miss"] == best],
        "minimum_max_fit_miss": best, "records": records, "targets": {},
    }
    for key, target in targets.items():
        if not accepted:
            break
        raw, transfer = [], []
        for record in accepted:
            value = runs[record["index"]]["predictions"][key]
            raw.append(value)
            # Keep the residual and prediction from the same physical run.
            for source in target["error_from"]:
                lo, hi = record["residual"][source]
                transfer.extend([value + lo, value + hi])
        sources = target["error_from"]
        result["targets"][key] = {
            "unit": target["unit"], "model_range": [min(raw), max(raw)],
            "residual_transfer_range": [min(transfer), max(transfer)] if transfer else None,
            "error_from": sources, "uses_held_out_data": any(
                observations[s]["role"] == "check" for s in sources
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = calibrate(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["accepted_indices"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
