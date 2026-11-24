"""Generate two synthetic buck power-stage coupons for placement comparison."""

import argparse
import json
from pathlib import Path

import pcbnew as k


def xy(x, y):
    return k.VECTOR2I(k.FromMM(x), k.FromMM(y))


def build(folder, cap_x):
    folder.mkdir(parents=True, exist_ok=True)
    board = k.BOARD()
    corners = [(0, 0), (28, 0), (28, 16), (0, 16)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        edge = k.PCB_SHAPE(board)
        edge.SetShape(k.SHAPE_T_SEGMENT)
        edge.SetLayer(k.Edge_Cuts)
        edge.SetStart(xy(*start))
        edge.SetEnd(xy(*end))
        edge.SetWidth(k.FromMM(0.05))
        board.Add(edge)

    nets = {}
    for name in ("SupplyRail", "ReferenceFoil", "PulseCopper", "FilteredRail"):
        nets[name] = k.NETINFO_ITEM(board, name)
        board.Add(nets[name])

    def component(ref, x, y, pads):
        part = k.FOOTPRINT(board)
        part.SetReference(ref)
        part.SetValue(ref)
        part.SetFPID(k.LIB_ID("Coupon", ref))
        part.SetPosition(xy(x, y))
        board.Add(part)
        for number, dx, dy, net in pads:
            pad = k.PAD(part)
            pad.SetNumber(number)
            pad.SetAttribute(k.PAD_ATTRIB_SMD)
            pad.SetShape(k.PAD_SHAPE_RECT)
            pad.SetLayerSet(pad.SMDMask())
            pad.SetSize(xy(1.3, 1.3))
            pad.SetPosition(xy(x + dx, y + dy))
            pad.SetNet(nets[net])
            part.Add(pad)

    def line(net, start, end, width, layer=k.F_Cu):
        track = k.PCB_TRACK(board)
        track.SetLayer(layer)
        track.SetWidth(k.FromMM(width))
        track.SetStart(xy(*start))
        track.SetEnd(xy(*end))
        track.SetNet(nets[net])
        board.Add(track)

    def barrel(x, y):
        via = k.PCB_VIA(board)
        via.SetPosition(xy(x, y))
        via.SetWidth(k.FromMM(0.8))
        via.SetDrill(k.FromMM(0.35))
        via.SetNet(nets["ReferenceFoil"])
        board.Add(via)

    pair = [("p", 0, -1.5, "SupplyRail"), ("n", 0, 1.5, "ReferenceFoil")]
    component("Inlet", 2, 8, pair)
    component("Bypass", cap_x, 8, pair)
    component("HalfBridge", 15, 8, pair + [("sw", 2, 0, "PulseCopper")])
    component("Winding", 20, 8, [("a", -1, 0, "PulseCopper"), ("b", 1, 0, "FilteredRail")])
    component("Reservoir", 25, 8, [("p", 0, 0, "FilteredRail"), ("n", 0, 3, "ReferenceFoil")])
    line("SupplyRail", (2, 6.5), (15, 6.5), 1.1)
    line("PulseCopper", (17, 8), (19, 8), 1.1)
    line("FilteredRail", (21, 8), (25, 8), 1.1)
    # The back copper is a wide return strip, connected by a short fanout at each device.
    line("ReferenceFoil", (2, 10.7), (25, 10.7), 4, k.B_Cu)
    for x in (2, cap_x, 15):
        line("ReferenceFoil", (x, 9.5), (x, 10.7), 0.8)
        barrel(x, 10.7)
    line("ReferenceFoil", (25, 11), (25, 12), 0.8)
    barrel(25, 12)
    k.SaveBoard(str(folder / "coupon.kicad_pcb"), board)
    terminals = lambda a, b: [{"contacts": [a], "current_A": 1}, {"contacts": [b], "current_A": -1}]
    legs = [["HalfBridge.p", "Bypass.p"], ["Bypass.n", "HalfBridge.n"]]
    cfg = {
        "material": {"depth_mm": {"F.Cu": 0, "B.Cu": 1.2}, "copper_mm": {"F.Cu": 0.035, "B.Cu": 0.035},
                     "plating_mm": 0.025, "rho_ohm_m": 1.72e-8},
        "electrical": [
            {"name": "Feed current", "net": "SupplyRail", "mesh_mm": 0.14,
             "ports": terminals("Inlet.p", "HalfBridge.p"), "frequency_Hz": [0]},
            {"name": "Capacitor return", "net": "ReferenceFoil", "mesh_mm": 0.14,
             "ports": terminals("HalfBridge.n", "Bypass.n"), "frequency_Hz": [1e6],
             "exciting_path_mm": [[cap_x, 9.5, 0], [cap_x, 6.5, 0], [15, 6.5, 0], [15, 9.5, 0]],
             "return_spacing_mm": 1.2, "display_mm": 0.28}],
        "thermal": {"Natural convection": {"ambient_c": 25, "material_mm": 0.5, "grid_mm": 0.25,
                    "part_power_w": {"HalfBridge": 0.18, "Winding": 0.055, "Bypass": 0.008},
                    "dc_cases": ["Feed current"]}},
        "pdn": [{"name": "Input impedance", "voltage_V": 8, "frequency_Hz": [10**(4+i/12) for i in range(49)],
                 "feed": {"resistance_ohm": 0.09, "inductance_H": 90e-9},
                 "capacitors": [{"name": "Input bypass", "legs": legs, "capacitance_F": 3.3e-6,
                                 "esr_ohm": 0.012, "esl_H": 0.6e-9, "bias_curve": [[0,1],[8,0.72]]}]}],
        "simulator": {"enabled": False},
    }
    (folder / "settings.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    base = parser.parse_args().directory
    for label, position in (("near", 12.5), ("far", 4.5)):
        build(base / label, position)
