"""Ordered routed centerlines, including arcs and the used length of through vias."""

from collections import defaultdict
import heapq
import math
import re

import numpy as np
from shapely.geometry import Point

from .mesh import polygon


def detect(names, suffixes=None):
    if suffixes is not None:
        if any(len(two) != 2 or not all(two) or two[0] == two[1] for two in suffixes):
            raise ValueError("Each rule needs distinct positive and negative suffixes")
        return [
            list(pair)
            for pair in sorted(
                {
                    (name, name[: -len(a)] + b)
                    for name in names
                    for a, b in suffixes
                    if name.endswith(a) and name[: -len(a)] + b in names
                }
            )
        ]
    pairs = []
    for name in sorted(names):
        match = re.fullmatch(r"(.*?)([_./-]P|\+)", name, re.I)
        if match:
            suffix = match[2]
            other = match[1] + (
                "-"
                if suffix == "+"
                else suffix[:-1] + ("n" if suffix[-1] == "p" else "N")
            )
            if other in names:
                pairs.append([name, other])
    return pairs


def point(edge, u):
    a, b = np.array(edge["a"]), np.array(edge["b"])
    if "mid" not in edge:
        return a + u * (b - a)
    m = np.array(edge["mid"])
    if np.linalg.norm(m - a) < 1e-10 or np.linalg.norm(m - b) < 1e-10:
        return a + u * (b - a)
    centre = np.linalg.solve(
        2 * np.stack([m - a, b - a]), [m @ m - a @ a, b @ b - a @ a]
    )
    angles = [math.atan2(*(p - centre)[::-1]) for p in (a, m, b)]
    sweep = (angles[2] - angles[0]) % (2 * math.pi)
    if (angles[1] - angles[0]) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    theta = angles[0] + u * sweep
    return centre + np.linalg.norm(a - centre) * np.array(
        [math.cos(theta), math.sin(theta)]
    )


def route(g, mat, name, ends=None):
    net = g["nets"][name]
    adj = defaultdict(list)

    def node(xy, l):
        return (round(xy[0], 6), round(xy[1], 6), l)

    def connect(i, j, edge):
        adj[i].append((j, edge))
        adj[j].append(
            (
                i,
                {
                    **edge,
                    "a": edge["b"],
                    "b": edge["a"],
                    "layer": j[2],
                    "to_layer": i[2],
                },
            )
        )

    for t in net["tracks"]:
        if t["length_mm"] > 0:
            connect(
                node(t["a"], t["layer"]),
                node(t["b"], t["layer"]),
                {**t, "kind": "track", "to_layer": t["layer"]},
            )
    depth = mat["depth_mm"]
    lands = defaultdict(list)
    pad_nodes = defaultdict(set)
    for pin, pads in net.get("pads", {}).items():
        for pad in pads:
            lands[pad["layer"]].append(
                (pin, node(pad["xy"], pad["layer"]), polygon(pad.get("polygons", [])))
            )

    def touching(n):
        position = Point(n[:2])
        return [
            pin
            for pin, center, shape in lands[n[2]]
            if center == n or shape.covers(position)
        ]

    for n in adj:
        for pin in touching(n):
            pad_nodes[pin].add(n)
    for v in net.get("vias", []):
        contacts = []
        for layer in sorted(v["layers"], key=depth.get):
            n = node(v["xy"], layer)
            pins = touching(n) if v.get("kind") != "plated_pad" else []
            if n in adj or pins:
                contacts.append(layer)
                for pin in pins:
                    pad_nodes[pin].add(n)
        for x, y in zip(contacts, contacts[1:]):
            connect(
                node(v["xy"], x),
                node(v["xy"], y),
                {
                    "kind": "via",
                    "a": v["xy"],
                    "b": v["xy"],
                    "layer": x,
                    "to_layer": y,
                    "width_mm": v["drill_mm"],
                    "length_mm": abs(depth[x] - depth[y]),
                    "via_xy": v["xy"],
                },
            )
    terminals = sorted(n for n, rows in adj.items() if len(rows) == 1)
    if ends:

        def terminal(end):
            if isinstance(end, str):
                hits = pad_nodes[end] & adj.keys()
            else:
                hits = {node(end["xy"], end["layer"])} & adj.keys()
            if len(hits) != 1:
                raise ValueError(
                    "Terminal must identify one routed endpoint and layer: " + str(end)
                )
            return hits.pop()

        start, end = map(terminal, ends)
    else:
        if len(terminals) != 2 or any(len(v) > 2 for v in adj.values()):
            raise ValueError("Set endpoints for a branched or ambiguous route: " + name)
        start, end = terminals
    if start == end:
        raise ValueError("Distinct route terminals required")
    dist = {start: 0}
    parent = {}
    queue = [(0, start)]
    while queue:
        d, i = heapq.heappop(queue)
        if d != dist[i]:
            continue
        if i == end:
            break
        for j, e in adj[i]:
            new = d + e["length_mm"]
            if new < dist.get(j, math.inf):
                dist[j] = new
                parent[j] = (i, e)
                heapq.heappush(queue, (new, j))
    if end not in parent:
        raise ValueError("Disconnected route: " + name)
    result = []
    cursor = end
    while cursor != start:
        cursor, edge = parent[cursor]
        result.append(edge)
    result.reverse()
    spans = defaultdict(set)
    for e in result:
        if e["kind"] == "via":
            spans[tuple(e["a"])].update((e["layer"], e["to_layer"]))
    bounds = (min(depth.values()), max(depth.values()))
    seen = set()
    for e in result:
        if e["kind"] == "via":
            at = tuple(e["a"])
            zs = [depth[l] for l in spans[at]]
            e["stubs_mm"] = (
                [min(zs) - bounds[0], bounds[1] - max(zs)] if at not in seen else []
            )
            seen.add(at)
    return result


def reverse(edges):
    return [
        {**e, "a": e["b"], "b": e["a"], "layer": e["to_layer"], "to_layer": e["layer"]}
        for e in reversed(edges)
    ]
