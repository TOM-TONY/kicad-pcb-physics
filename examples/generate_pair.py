"""Generate a fictional differential link with a jog and a split reference foil."""

import argparse
import json
from pathlib import Path
import pcbnew as k


def xy(x, y):
    return k.VECTOR2I(k.FromMM(x), k.FromMM(y))


def build(folder):
    folder.mkdir(parents=True, exist_ok=True)
    board = k.BOARD()
    for a, b in zip(
        [(0, 0), (40, 0), (40, 18), (0, 18)], [(40, 0), (40, 18), (0, 18), (0, 0)]
    ):
        e = k.PCB_SHAPE(board)
        e.SetShape(k.SHAPE_T_SEGMENT)
        e.SetLayer(k.Edge_Cuts)
        e.SetStart(xy(*a))
        e.SetEnd(xy(*b))
        e.SetWidth(k.FromMM(0.05))
        board.Add(e)

    def net(name):
        n = k.NETINFO_ITEM(board, name)
        board.Add(n)
        return n

    def track(n, layer, width, points):
        for a, b in zip(points, points[1:]):
            t = k.PCB_TRACK(board)
            t.SetLayer(layer)
            t.SetNet(n)
            t.SetWidth(k.FromMM(width))
            t.SetStart(xy(*a))
            t.SetEnd(xy(*b))
            board.Add(t)

    def foil(n, layer, x0, y0, x1, y1):
        zone = k.ZONE(board)
        zone.SetLayer(layer)
        zone.SetNet(n)
        zone.Outline().NewOutline()
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            zone.Outline().Append(xy(x, y))
        zone.SetIslandRemovalMode(k.ISLAND_REMOVAL_MODE_NEVER)
        board.Add(zone)

    g = net("DatumFilm")
    p = net("Ribbon_P")
    n = net("Ribbon_N")
    track(
        p,
        k.F_Cu,
        0.356,
        [(3, 7.5), (12, 7.5), (14, 5.5), (18, 5.5), (20, 7.5), (37, 7.5)],
    )
    track(n, k.F_Cu, 0.356, [(3, 8.16), (37, 8.16)])
    foil(g, k.B_Cu, 0.5, 0.5, 25, 17.5)
    foil(g, k.B_Cu, 26.5, 0.5, 39.5, 17.5)
    track(g, k.B_Cu, 1, [(23, 15), (28, 15)])
    foil(g, k.F_Cu, 1, 11, 39, 17)
    for x, y in [(2, 14), (37, 14)]:
        v = k.PCB_VIA(board)
        v.SetPosition(xy(x, y))
        v.SetWidth(k.FromMM(0.65))
        v.SetDrill(k.FromMM(0.30))
        v.SetNet(g)
        board.Add(v)
    k.ZONE_FILLER(board).Fill(board.Zones())
    k.SaveBoard(str(folder / "pair.kicad_pcb"), board)
    cfg = {
        "material": {
            "depth_mm": {"F.Cu": 0, "B.Cu": 0.20},
            "copper_mm": {"F.Cu": 0.018, "B.Cu": 0.018},
            "rho_ohm_m": 1.72e-8,
            "plating_mm": 0.02,
        },
        "transmission": {
            "reference": "DatumFilm",
            "detect_pairs": True,
            "frequency_hz": [1e7, 1e8, 5e8, 1e9, 2e9],
            "stack": {
                "F.Cu": {
                    "plane": "B.Cu",
                    "er": 4.2,
                    "tan_delta": 0.015,
                    "mask_mm": 0.015,
                    "mask_er": 3.3,
                    "mask_tan_delta": 0.025,
                }
            },
            "roughness_um": 1,
            "step_mm": 0.4,
            "cells": 24,
            "margin": 18,
            "design": {
                "single_ohm": 50,
                "diff_ohm": 90,
                "width_range_mm": [0.21, 0.65],
                "gap_range_mm": [0.04, 0.60],
                "loss_db": 1.0,
                "skew_ps": 10,
            },
        },
        "stitching": [
            {
                "name": "Foil connection",
                "net": "DatumFilm",
                "objective": "resistance",
                "source": {"layer": "F.Cu", "xy": [16, 13.8], "radius_mm": 0.45},
                "sink": {"layer": "B.Cu", "xy": [16, 13.8], "radius_mm": 0.45},
                "current_A": 1,
                "frequency_hz": 0,
                "max_frequency_hz": 2e9,
                "er": 4.2,
                "mesh_mm": 0.4,
                "search_step_mm": 1,
                "max_vias": 2,
                "land_mm": 0.65,
                "drill_mm": 0.30,
                "clearance_mm": 0.2,
                "hole_clearance_mm": 0.25,
                "edge_clearance_mm": 0.35,
                "via_spacing_mm": 1.1,
            }
        ],
    }
    (folder / "settings.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
