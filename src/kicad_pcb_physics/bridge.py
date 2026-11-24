"""Standalone filled-board geometry exporter. Run with the system KiCad interpreter."""

import argparse
import json
import math
from pathlib import Path
import pcbnew as k


def xy(point):
    return [round(k.ToMM(point.x), 6), round(k.ToMM(point.y), 6)]


def ring(chain):
    return [xy(chain.CPoint(i)) for i in range(chain.PointCount())]


def polygons(poly):
    return [
        {
            "shell": ring(poly.COutline(i)),
            "holes": [ring(poly.CHole(i, j)) for j in range(poly.HoleCount(i))],
        }
        for i in range(poly.OutlineCount())
    ]


def shape(item, layer):
    p = k.SHAPE_POLY_SET()
    item.TransformShapeToPolygon(p, layer, 0, 1000, k.ERROR_INSIDE)
    return p


def export(path):
    board = k.LoadBoard(str(path))
    layers = list(board.GetEnabledLayers().CuStack())
    names = [board.GetLayerName(layer) for layer in layers]
    nets, parts, unions = {}, {}, {}
    pad_obstacles = []
    keepouts = []
    extra_metal = {l: k.SHAPE_POLY_SET() for l in layers}
    holes = k.SHAPE_POLY_SET()
    profile = k.SHAPE_POLY_SET()
    if not board.GetBoardPolygonOutlines(profile, False):
        raise ValueError("Board needs a closed outline")

    def net(name):
        if name not in nets:
            nets[name] = {"copper": {}, "tracks": [], "vias": [], "pads": {}}
            unions[name] = {layer: k.SHAPE_POLY_SET() for layer in layers}
        return nets[name]

    def through(name, at, drill, land, span, kind="via"):
        net(name)["vias"].append(
            {
                "xy": xy(at),
                "drill_mm": drill,
                "kind": kind,
                "land_mm": land,
                "layers": [board.GetLayerName(l) for l in span],
            }
        )

    for f in board.GetFootprints():
        ref = f.GetReference()
        bb = f.GetBoundingBox(False, False)
        mounted = board.GetLayerName(f.GetLayer())
        parts[ref] = {
            "xy": xy(f.GetPosition()),
            "layer": mounted,
            "bounds": [
                k.ToMM(v)
                for v in (bb.GetLeft(), bb.GetTop(), bb.GetRight(), bb.GetBottom())
            ],
        }
        for p in f.Pads():
            name = p.GetNetname()
            drill = p.GetDrillSize()
            if drill.x:
                bore = k.SHAPE_POLY_SET()
                p.TransformHoleToPolygon(bore, 0, 1000, k.ERROR_INSIDE)
                holes.Append(bore)
            for l in layers:
                if p.IsOnLayer(l):
                    solid = shape(p, l)
                    pad_obstacles.append(
                        {"layer": board.GetLayerName(l), "polygons": polygons(solid)}
                    )
                    if not name:
                        extra_metal[l].Append(solid)
            if not name:
                continue
            data = net(name)
            key = ref + "." + p.GetNumber()
            span = [l for l in layers if p.IsOnLayer(l) and p.FlashLayer(l)]
            for l in span:
                metal = shape(p, l)
                unions[name][l].Append(metal)
                if drill.x:
                    metal.BooleanSubtract(bore)
                data["pads"].setdefault(key, []).append(
                    {
                        "xy": xy(p.GetPosition()),
                        "layer": board.GetLayerName(l),
                        "polygons": polygons(metal),
                    }
                )
            if drill.x and p.GetAttribute() == k.PAD_ATTRIB_PTH:
                # An equivalent circular perimeter also handles capsule-shaped slots.
                a, b = sorted(xy(drill))
                through(
                    name,
                    p.GetPosition(),
                    a + 2 * (b - a) / math.pi,
                    min(xy(p.GetSize())),
                    span,
                    "plated_pad",
                )
    for t in board.GetTracks():
        name = t.GetNetname()
        if not name:
            for l in layers:
                if t.IsOnLayer(l):
                    extra_metal[l].Append(shape(t, l))
            if isinstance(t, k.PCB_VIA):
                bore = k.SHAPE_POLY_SET()
                k.SHAPE_CIRCLE(
                    t.GetPosition(), t.GetDrillValue() // 2
                ).TransformToPolygon(bore, 1000, k.ERROR_OUTSIDE)
                holes.Append(bore)
            continue
        data = net(name)
        if isinstance(t, k.PCB_VIA):
            if t.GetViaType() != k.VIATYPE_THROUGH:
                raise ValueError("Blind and buried vias are unsupported")
            span = [l for l in layers if t.FlashLayer(l)]
            through(
                name,
                t.GetPosition(),
                k.ToMM(t.GetDrillValue()),
                k.ToMM(t.GetWidth(layers[0])),
                span,
            )
            circ = k.SHAPE_POLY_SET()
            k.SHAPE_CIRCLE(t.GetPosition(), t.GetDrillValue() // 2).TransformToPolygon(
                circ, 1000, k.ERROR_INSIDE
            )
            holes.Append(circ)
            for l in span:
                unions[name][l].Append(shape(t, l))
        else:
            l = t.GetLayer()
            unions[name][l].Append(shape(t, l))
            track = {
                "a": xy(t.GetStart()),
                "b": xy(t.GetEnd()),
                "width_mm": k.ToMM(t.GetWidth()),
                "layer": board.GetLayerName(l),
                "length_mm": k.ToMM(t.GetLength()),
            }
            if isinstance(t, k.PCB_ARC):
                track["mid"] = xy(t.GetMid())
            data["tracks"].append(track)
    artwork = list(board.GetDrawings())
    zones = list(board.Zones())
    for fp in board.GetFootprints():
        artwork.extend(fp.GraphicalItems())
        artwork.extend(fp.GetFields())
        zones.extend(fp.Zones())
    for item in artwork:
        for l in layers:
            if item.IsOnLayer(l):
                extra_metal[l].Append(shape(item, l))
    for zone in zones:
        if zone.GetIsRuleArea() and zone.GetDoNotAllowVias():
            keepouts.append(
                {
                    "layers": [
                        board.GetLayerName(l) for l in layers if zone.IsOnLayer(l)
                    ],
                    "polygons": polygons(zone.Outline()),
                }
            )
        if zone.GetIsRuleArea():
            continue
        name = zone.GetNetname()
        if name:
            net(name)
        target = unions[name] if name else extra_metal
        for l in layers:
            if zone.IsOnLayer(l):
                if not zone.HasFilledPolysForLayer(l):
                    raise ValueError(
                        "Unfilled zone: refill the input board in its editor"
                    )
                target[l].Append(zone.GetFilledPolysList(l))
    holes.Simplify()
    for name, layerset in unions.items():
        for l, metal in layerset.items():
            metal.Simplify()
            if metal.OutlineCount() and holes.OutlineCount():
                metal.BooleanSubtract(holes)
            nets[name]["copper"][board.GetLayerName(l)] = polygons(metal)
    return {
        "layers": names,
        "outline": polygons(profile),
        "holes": polygons(holes),
        "pad_obstacles": pad_obstacles,
        "via_keepouts": keepouts,
        "extra_metal": {
            board.GetLayerName(l): polygons(v) for l, v in extra_metal.items()
        },
        "nets": nets,
        "parts": parts,
        "exporter_version": k.Version(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    if args.destination.suffix.lower() != ".json" or (
        args.destination.exists() and args.board.samefile(args.destination)
    ):
        p.error("Choose a separate JSON output path")
    args.destination.write_text(json.dumps(export(args.board), separators=(",", ":")))
