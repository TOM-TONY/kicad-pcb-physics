"""Generate a synthetic two-layer strip fixture and stimuli; uses the KiCad interpreter."""

import argparse
import json
import math
from pathlib import Path
import pcbnew as k


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    board = k.BOARD()
    vec = lambda x, y: k.VECTOR2I(k.FromMM(x), k.FromMM(y))
    for a, b in zip(
        [(0, 0), (24, 0), (24, 12), (0, 12)], [(24, 0), (24, 12), (0, 12), (0, 0)]
    ):
        edge = k.PCB_SHAPE(board)
        edge.SetShape(k.SHAPE_T_SEGMENT)
        edge.SetLayer(k.Edge_Cuts)
        edge.SetStart(vec(*a))
        edge.SetEnd(vec(*b))
        edge.SetWidth(k.FromMM(0.05))
        board.Add(edge)
    pins = {}
    for name, y in (("FORWARD", 4), ("RETURN", 6)):
        net = k.NETINFO_ITEM(board, name)
        board.Add(net)
        for ref, x in (("PORTX", 3), ("DISC", 14), ("ENDX", 21)):
            if ref not in pins:
                f = k.FOOTPRINT(board)
                f.SetFPID(k.LIB_ID("Fixture", ref))
                f.SetReference(ref)
                f.SetValue("Probe")
                f.SetPosition(vec(x, 5))
                board.Add(f)
                pins[ref] = f
            pad = k.PAD(pins[ref])
            pad.SetNumber("p" if y == 4 else "n")
            pad.SetShape(k.PAD_SHAPE_RECT)
            pad.SetAttribute(k.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pad.SMDMask())
            pad.SetSize(vec(1.2, 1.2))
            pad.SetPosition(vec(x, y))
            pad.SetNet(net)
            pins[ref].Add(pad)
        t = k.PCB_TRACK(board)
        t.SetStart(vec(3, y))
        t.SetEnd(vec(21, y))
        t.SetWidth(k.FromMM(1))
        t.SetLayer(k.F_Cu)
        t.SetNet(net)
        board.Add(t)
        if name == "RETURN":
            for x in (3, 21):
                via = k.PCB_VIA(board)
                via.SetPosition(vec(x, 6))
                via.SetWidth(k.FromMM(1.2))
                via.SetDrill(k.FromMM(0.5))
                via.SetNet(net)
                board.Add(via)
            back = [(3, 6), (3, 4), (9, 4), (9, 8), (14, 8), (14, 4), (21, 4), (21, 6)]
            for start, end in zip(back, back[1:]):
                track = k.PCB_TRACK(board)
                track.SetStart(vec(*start))
                track.SetEnd(vec(*end))
                track.SetWidth(k.FromMM(1.5))
                track.SetLayer(k.B_Cu)
                track.SetNet(net)
                board.Add(track)
    k.SaveBoard(str(folder / "fixture.kicad_pcb"), board)
    f = [10 ** (3 + i * 0.2) for i in range(31)]
    ports = lambda p: [
        {"contacts": ["PORTX." + p], "current_A": 1},
        {"contacts": ["ENDX." + p], "current_A": -1},
    ]
    legs = [["PORTX.p", "DISC.p"], ["DISC.n", "PORTX.n"]]
    settings = {
        "material": {
            "depth_mm": {"F.Cu": 0, "B.Cu": 1.5},
            "copper_mm": {"F.Cu": 0.035, "B.Cu": 0.035},
            "plating_mm": 0.025,
            "rho_ohm_m": 1.72e-8,
        },
        "electrical": [
            {
                "name": "Strip drive",
                "net": "FORWARD",
                "mesh_mm": 0.16,
                "ports": ports("p"),
                "frequency_Hz": [0, 1e5, 1e7],
                "field": {
                    "frequency_Hz": 1e7,
                    "x_mm": list(range(2, 23)),
                    "y_mm": list(range(1, 12)),
                    "z_mm": -1.5,
                    "image_z_mm": 1.5,
                    "charge_nodes": [
                        {"xyz_mm": [10, 4, 0], "voltage_V": 0.03, "cap_F": 2e-12}
                    ],
                },
            },
            {
                "name": "Return response",
                "net": "RETURN",
                "mesh_mm": 0.2,
                "ports": ports("n"),
                "frequency_Hz": [1e5, 1e7],
                "return_spacing_mm": 1.5,
                "radiation_distance_m": 30,
                "exciting_path_mm": [[21, 4, 0], [3, 4, 0]],
            },
        ],
        "thermal": {
            "Air cooled": {
                "grid_mm": 0.5,
                "material_mm": 1,
                "ambient_c": 25,
                "part_power_w": {"DISC": 0.1},
                "dc_cases": ["Strip drive"],
            }
        },
        "pdn": [
            {
                "name": "Port sweep",
                "voltage_V": 2.0,
                "frequency_Hz": f,
                "feed": {"resistance_ohm": 0.08, "inductance_H": 25e-9},
                "capacitors": [
                    {
                        "name": "Local",
                        "legs": legs,
                        "capacitance_F": 1e-6,
                        "bias_curve": [[0, 1], [4, 0.75]],
                        "esr_ohm": 0.015,
                        "esl_H": 0.6e-9,
                    }
                ],
                "periodic_current": {
                    "dt_s": 2e-9,
                    "samples_A": [
                        0.25 + 0.2 * math.sin(2 * math.pi * i / 128) for i in range(128)
                    ],
                },
            }
        ],
        "returns": [
            {
                "name": "Opposite-face coverage",
                "signal": "FORWARD",
                "reference": "RETURN",
                "layer": "F.Cu",
                "reference_layer": "B.Cu",
            }
        ],
        "receivers": [
            {
                "name": "Uniform exposure",
                "frequency_Hz": 1e6,
                "E_V_m": [2, 0, 0],
                "B_T": [0, 0, 1e-7],
                "paths": [
                    {
                        "name": "Strip terminals",
                        "legs": legs,
                        "harness_V": [0.002, 0],
                        "source_ohm": [50, 0],
                        "load_ohm": [50, 0],
                    }
                ],
            }
        ],
        "radiation": [
            {
                "name": "Loop emission estimate",
                "pdn_case": "Port sweep",
                "branch": 0,
                "distance_m": 10,
                "current_A": 1,
                "frequency_Hz": [1e7, 5e7],
            }
        ],
    }
    settings["exposure"] = [
        {
            "name": "Terminal pickup",
            "frequency_Hz": 2e6,
            "reference_depth_mm": 1.6,
            "reference_xy_mm": [3, 6],
            "E_V_m": [1, 0.2, 0],
            "B_T": [0, 1e-7, 1e-7],
        }
    ]
    settings["electrical"][0]["field"]["sheet_charge"] = {
        "image_z_mm": 1.5,
        "epsilon_r": 4,
        "common_mode_V": 0.01,
    }
    from copy import deepcopy

    extra = deepcopy(settings["pdn"][0])
    extra["name"] = "Second branch"
    extra["capacitors"].append(
        {
            "name": "Far",
            "legs": [["PORTX.p", "ENDX.p"], ["ENDX.n", "PORTX.n"]],
            "capacitance_F": 4.7e-6,
            "esr_ohm": 0.025,
            "esl_H": 1.2e-9,
        }
    )
    settings["pdn"].append(extra)
    (folder / "charge.cir").write_text(
        "Charging coupon\nVstep source 0 1\nRcharge source out @@ohms@@\nCstore out 0 1u IC=0\n.tran 1u 2m uic\n.meas tran received FIND V(out) AT=1m\n.end\n"
    )
    settings["transients"] = [
        {
            "name": "Charge response",
            "template": "charge.cir",
            "parameters": {"ohms": 1000},
            "measurements": [{"name": "received", "unit": "V", "parts": ["DISC"]}],
        }
    ]
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    generate(p.parse_args().directory)
