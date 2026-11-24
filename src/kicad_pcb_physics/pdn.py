"""Small-signal port impedance and periodic current response, including mutual inductance."""

import numpy as np
from . import paths
from .mesh import Sheet


def biased(cap, voltage):
    value = cap["capacitance_F"]
    if "bias_curve" in cap:
        curve = np.array(cap["bias_curve"])
        if (
            np.any(np.diff(curve[:, 0]) <= 0)
            or not curve[0, 0] <= voltage <= curve[-1, 0]
        ):
            raise ValueError(
                "Bias must lie inside a strictly ordered capacitance curve"
            )
        value *= np.interp(abs(voltage), curve[:, 0], curve[:, 1])
    if value <= 0:
        raise ValueError("Capacitance must be positive")
    return value


def copper_transfer(geo, material, prepared, parts, cfg):
    """Solve each driven branch with all other finite electrodes open and equipotential."""
    n = len(parts)
    matrix = np.zeros((n, n))
    grouped = {}
    for i, cap in enumerate(parts):
        for a, b in cap["legs"]:
            net = prepared["ports"][a]["net"]
            grouped.setdefault(net, []).append((i, a, b))
    for net, legs in grouped.items():
        pins = sorted({p for _, a, b in legs for p in (a, b)})
        locations = np.array([prepared["ports"][p]["xy"] for p in pins])
        margin = cfg.get("resistance_margin_mm", cfg["roi_margin_mm"])
        bounds = (*(locations.min(axis=0) - margin), *(locations.max(axis=0) + margin))
        sheet = Sheet(
            geo,
            material,
            net,
            cfg.get("resistance_pitch_mm", max(0.12, cfg["pitch_mm"])),
            bounds,
        )
        excitations = np.zeros((n, len(pins)))
        for branch, a, b in legs:
            excitations[branch, pins.index(a)] += 1
            excitations[branch, pins.index(b)] -= 1
        for j, amps in enumerate(excitations):
            if not amps.any():
                continue
            ports = [
                {"contacts": [pin], "current_A": amps[k]} for k, pin in enumerate(pins)
            ]
            _, voltages, *_ = sheet.solve(ports)
            matrix[:, j] += excitations @ voltages
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-11):
        raise ArithmeticError("Copper transfer violates reciprocity")
    return (matrix + matrix.T) / 2


def impedance(frequencies, capacitance, resistance, inductance, feed):
    L = np.asarray(inductance, float)
    if np.any(np.linalg.eigvalsh(L) <= 0):
        raise ValueError("Non-passive inductance matrix; refine paths or separation")
    C, R = np.asarray(capacitance), np.asarray(resistance)
    if R.ndim == 1:
        R = np.diag(R)
    result = []
    for f in frequencies:
        if f == 0:
            result.append(complex(feed["resistance_ohm"]))
            continue
        s = 2j * np.pi * f
        block = R + np.diag(1 / (s * C)) + s * L
        y = sum(np.linalg.solve(block, np.ones(len(C))))
        result.append(1 / (y + 1 / (feed["resistance_ohm"] + s * feed["inductance_H"])))
    return np.array(result)


def analyze(geo, material, spec):
    cfg = {
        "pitch_mm": 0.1,
        "roi_margin_mm": 8.0,
        "centerline_penalty_mm": 0.04,
        "width_cap_mm": 2.0,
        "segment_mm": 0.12,
        **spec.get("routing", {}),
    }
    all_legs = [leg for cap in spec["capacitors"] for leg in cap["legs"]]
    prepared = paths.prepare(geo, material, all_legs)
    loops = [
        paths.loop(prepared, material, cap["legs"], cfg) for cap in spec["capacitors"]
    ]
    parts = spec["capacitors"]
    C = [biased(c, spec["voltage_V"]) for c in parts]
    copper = copper_transfer(geo, material, prepared, parts, cfg)
    R = copper + np.diag([c["esr_ohm"] for c in parts])
    L = np.diag([u["inductance_nH"] * 1e-9 + c["esl_H"] for u, c in zip(loops, parts)])
    for i, u in enumerate(loops):
        for j, v in enumerate(loops[:i]):
            L[i, j] = L[j, i] = paths.mutual_nh(u, v, cfg["segment_mm"]) * 1e-9
    f = np.asarray(spec["frequency_Hz"])
    Z = impedance(f, C, R, L, spec["feed"])
    out = {
        "name": spec["name"],
        "loops": loops,
        "effective_F": C,
        "branch_R_ohm": np.diag(R).tolist(),
        "resistance_matrix_ohm": R.tolist(),
        "copper_transfer_ohm": copper.tolist(),
        "inductance_matrix_H": L.tolist(),
        "frequency_Hz": f.tolist(),
        "z_abs_ohm": abs(Z).tolist(),
        "phase_deg": np.angle(Z, deg=True).tolist(),
        "maps": [],
    }
    if "periodic_current" in spec:
        drive = spec["periodic_current"]
        wave = np.asarray(drive["samples_A"], float)
        frequencies = np.fft.rfftfreq(len(wave), drive["dt_s"])
        spectrum = impedance(frequencies, C, R, L, spec["feed"]) * np.fft.rfft(
            wave - wave.mean()
        )
        ripple = -np.fft.irfft(spectrum, n=len(wave))
        out.update(
            ripple_pp_V=float(np.ptp(ripple)),
            ripple_V=ripple.tolist(),
            dt_s=drive["dt_s"],
        )
    for cap, item in zip(parts, loops):
        for i, layer in enumerate(geo["layers"]):
            points = [
                [v[0], v[1], item["inductance_nH"]]
                for v in item["path_mm"]
                if abs(v[2] - material["depth_mm"][layer]) < 1e-6
            ]
            if points:
                out["maps"].append(
                    {
                        "name": spec["name"] + " / " + cap["name"] + " loop L",
                        "layer": layer,
                        "unit": "nH",
                        "points": points,
                    }
                )
    port = prepared["ports"][parts[0]["legs"][0][0]]
    layer = geo["layers"][port["layer"]]

    def port_map(label, unit, value):
        out["maps"].append(
            {
                "name": spec["name"] + " / " + label,
                "layer": layer,
                "unit": unit,
                "points": [[*port["xy"], float(value)]],
            }
        )

    if "ripple_pp_V" in out:
        port_map("supply ripple", "V pp", out["ripple_pp_V"])
    selected = spec.get(
        "map_frequencies_Hz", [float(f[0]), float(f[len(f) // 2]), float(f[-1])]
    )
    for freq in selected:
        if freq <= 0:
            raise ValueError("Positive overlay frequency required")
        omega = 2j * np.pi * freq
        zport = impedance([freq], C, R, L, spec["feed"])[0]
        branch = np.linalg.solve(
            R + np.diag(1 / (omega * np.array(C))) + omega * L, np.full(len(C), zport)
        )
        port_map(f"{freq:g} Hz port magnitude", "ohm", abs(zport))
        for cap, item, current in zip(parts, loops, branch):
            for copper in geo["layers"]:
                pts = [
                    [*pt[:2], float(abs(current))]
                    for pt in item["path_mm"]
                    if abs(pt[2] - material["depth_mm"][copper]) < 1e-7
                ]
                if pts:
                    out["maps"].append(
                        {
                            "name": f"{spec['name']} / {cap['name']} {freq:g} Hz branch",
                            "unit": "A per A",
                            "layer": copper,
                            "points": pts,
                        }
                    )
    return out
