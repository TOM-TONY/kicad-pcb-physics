"""Synthetic first-order thermal experiment with a separate cooling observation."""

import argparse
import json
from math import exp
from pathlib import Path


def rise(power, seconds, resistance, capacity):
    return power * resistance * (1 - exp(-seconds / (resistance * capacity)))


def experiment():
    def response(resistance, capacity):
        warm = rise(0.8, 90, resistance, capacity)
        return {"warm": warm, "steady": 0.8 * resistance,
                "cool": warm * exp(-30 / (resistance * capacity)),
                "pulse": rise(1.2, 60, resistance, capacity)}

    truth = response(32, 1.5)
    observations = [
        {"id": key, "low": truth[key] - 0.4, "high": truth[key] + 0.4,
         "scale": 0.4, "role": role, "unit": "K"}
        for key, role in (("warm", "fit"), ("steady", "fit"), ("cool", "check"))
    ]
    runs = [{"parameters": {"resistance": r, "capacity": c}, "predictions": response(r, c)}
            for r in (28, 30, 32, 34, 36) for c in (1.25, 1.5, 1.75, 2.0)]
    return {"observations": observations, "runs": runs,
            "targets": [{"id": "pulse", "unit": "K", "error_from": ["cool"]}]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.parse_args().output.write_text(json.dumps(experiment(), indent=2) + "\n")
