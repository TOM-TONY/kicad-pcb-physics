"""Write a new, explicitly requested KiCad proposal copy with through vias."""

import json
import os
from pathlib import Path
import sys
import tempfile

import pcbnew as k


def write(board, destination, spec):
    if (
        board.resolve() == destination.resolve()
        or destination.suffix != ".kicad_pcb"
        or destination.exists()
    ):
        raise ValueError("Choose a new .kicad_pcb file for the proposal")
    design = k.LoadBoard(str(board))
    net = design.FindNet(spec["net"])
    if not net:
        raise ValueError("Unknown via net")
    for x, y in spec["coordinates"]:
        via = k.PCB_VIA(design)
        via.SetViaType(k.VIATYPE_THROUGH)
        via.SetPosition(k.VECTOR2I(k.FromMM(x), k.FromMM(y)))
        via.SetWidth(k.FromMM(spec["land_mm"]))
        via.SetDrill(k.FromMM(spec["drill_mm"]))
        via.SetNet(net)
        design.Add(via)
    k.ZONE_FILLER(design).Fill(design.Zones())
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        out = Path(tmp) / destination.name
        if not k.SaveBoard(str(out), design):
            raise OSError("Could not save board copy")
        os.link(out, destination)


if __name__ == "__main__":
    write(
        Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
    )
