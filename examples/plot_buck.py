"""Render the output of both generated buck studies as a standalone SVG."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def render(folder, output):
    runs = [json.loads((folder / case / "result/results.json").read_text()) for case in ("near", "far")]
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox="0 0 980 555")
    ET.SubElement(root, "title").text = "Input bypass placement on a two-layer buck coupon"
    ET.SubElement(root, "desc").text = "Two impedance curves and paired maps of temperature and bottom copper return current."

    def rect(x, y, w, h, fill):
        ET.SubElement(root, "rect", x=f"{x:.2f}", y=f"{y:.2f}", width=f"{w:.2f}", height=f"{h:.2f}", fill=fill)

    def label(x, y, message, size=14):
        ET.SubElement(root, "text", x=str(x), y=str(y), fill="#172d3a",
                      **{"font-family": "sans-serif", "font-size": str(size)}).text = message

    rect(0, 0, 980, 555, "white")
    label(25, 32, "Input bypass placement", 24)
    label(25, 57, "Identical losses and material assumptions in both layouts")
    label(25, 103, "Input-port impedance", 18)
    left, top, w, h = 60, 140, 245, 270
    for power in range(-2, 2):
        y = top + h * (1 - power) / 3
        rect(left, y, w, 0.7, "#bcc8cd")
        label(12, y + 4, f"{10**power:g}")
    for power in range(4, 9):
        x = left + w * (power - 4) / 4
        rect(x, top, 0.7, h, "#bcc8cd")
        label(x - 12, top + h + 22, f"1e{power}", 12)
    label(25, 128, "|Z| (ohm)")
    label(123, 456, "Frequency (Hz)")
    for r, caption, color, y in zip(runs, ("Near", "Far"), ("#086978", "#b34c22"), (482, 508)):
        port = r["pdn"][0]
        pts = [(left + w * (math.log10(f) - 4) / 4, top + h * (1 - math.log10(z)) / 3)
               for f, z in zip(port["frequency_Hz"], port["z_abs_ohm"])]
        ET.SubElement(root, "polyline", points=" ".join(f"{x:.2f},{v:.2f}" for x,v in pts),
                      fill="none", stroke=color, **{"stroke-width":"2.6"})
        rect(38, y - 8, 29, 3, color)
        label(75, y, f"{caption}   {port['inductance_matrix_H'][0][0]*1e9:.2f} nH")
    for idx, (family, unit, layer, heading, pitch) in enumerate([
        ("thermal", "degC", "F.Cu", "Top surface temperature", 0.25),
        ("electrical", "A/mm2", "B.Cu", "Return density at 1 MHz", 1.0),
    ]):
        data = []
        for r in runs:
            candidates = r[family] if family == "thermal" else [r[family][1]]
            data.append(next(m["points"] for c in candidates for m in c["maps"]
                             if m["layer"] == layer and m["unit"] == unit))
        # Map bin size is inferred from the exported grid rather than from the display zoom.
        xs = sorted(set(p[0] for p in data[0]))
        pitch = min(b-a for a,b in zip(xs,xs[1:]) if b-a > 1e-6)
        values = [p[2] for points in data for p in points]
        lo, hi = (min(values), max(values)) if family == "thermal" else (0, max(values))
        origin = 348 + idx * 314
        label(origin, 103, heading, 18)

        def shade(value):
            t = max(0, min(1, (value-lo)/(hi-lo)))
            colors = [(233,245,243),(71,168,156),(15,65,92)] if idx else [(255,245,217),(243,163,83),(146,42,59)]
            i = min(1, int(2*t))
            return "#%02x%02x%02x" % tuple(round(a+(b-a)*(2*t-i)) for a,b in zip(colors[i],colors[i+1]))

        for j, pts in enumerate(data):
            y0 = 154 + 194*j
            label(origin, y0-12, ["Near","Far"][j])
            rect(origin, y0, 280, 160, "#eff2f4")
            for px,py,val in pts:
                rect(origin+(px-pitch/2)*10, y0+(py-pitch/2)*10, pitch*10+0.02, pitch*10+0.02, shade(val))
        for i in range(140):
            rect(origin+2*i, 521, 2.05, 6, shade(lo + i*(hi-lo)/139))
        label(origin, 548, f"{lo:.2f} to {hi:.2f} {'deg C' if idx == 0 else 'A/mm²'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.directory, args.output)
