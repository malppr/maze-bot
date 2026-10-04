# ruff: noqa: E501
"""Which mazes does a policy fail on? Per-map difficulty features vs success, compared with a baseline.

uv run python scripts/analyze_mazes.py artifacts/runs/<run>/eval [--baseline artifacts/eval/reactive] [--category mazes]
Needs episodes.jsonl from `mazebot.evaluate` for both. Writes <eval dir>/analysis_<category>.{md,png}.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mazebot import geometry
from mazebot.evaluate import eval_map

ap = argparse.ArgumentParser()
ap.add_argument("eval_dir")
ap.add_argument("--baseline", default="artifacts/eval/reactive")
ap.add_argument("--category", default="mazes")
args = ap.parse_args()


def load(d):
    eps = [json.loads(line) for line in open(Path(d) / "episodes.jsonl", encoding="utf-8")]
    return {e["index"]: e for e in eps if e["category"] == args.category}


ours, base = load(args.eval_dir), load(args.baseline)
idx = sorted(set(ours) & set(base))


def path_features(m):
    """Trace the shortest path on the grid (steepest descent of the geodesic field) from A to B."""
    g, f = m.grid, m.field
    ny, nx = f.shape
    i, j = g.index(*m.start[:2])
    # start from the best reachable cell near A
    cand = [
        (f[b, a], a, b)
        for b in range(max(0, j - 2), min(ny, j + 3))
        for a in range(max(0, i - 2), min(nx, i + 3))
    ]
    _, i, j = min(cand)
    pts, dirs = [g.centre(i, j)], []
    for _ in range(nx * ny):
        best = (f[j, i], 0, 0)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < nx and 0 <= b < ny and f[b, a] < best[0]:
                    best = (f[b, a], di, dj)
        if best[1] == 0 and best[2] == 0:
            break
        i, j = i + best[1], j + best[2]
        pts.append(g.centre(i, j))
        dirs.append(math.atan2(best[2], best[1]))
    pts = np.array(pts)
    # turns: heading changes > 60 deg after smoothing over ~0.6 units
    k = 4
    turns = 0
    if len(pts) > 2 * k:
        v = pts[k:] - pts[:-k]
        ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        acc = 0.0
        for d in np.diff(ang):
            acc += d
            if abs(acc) > math.radians(60):
                turns += 1
                acc = 0.0
    clear = g.clear[[g.index(x, y)[1] for x, y in pts], [g.index(x, y)[0] for x, y in pts]]
    return turns, float(np.percentile(clear, 10))


rows = []
for i in idx:
    m = eval_map(args.category, i)
    sx, sy, _ = m.start
    gx, gy = m.goal
    euclid = math.hypot(gx - sx, gy - sy)
    geo = ours[i]["geo0"]
    d = np.array([[gx - sx, gy - sy]]) / max(euclid, 1e-9)
    visible = geometry.cast_rays(sx, sy, d, m.caps, euclid + 1)[0] >= euclid - 0.35
    turns, narrow = path_features(m)
    rows.append(
        dict(
            index=i,
            ours=ours[i]["success"],
            base=base[i]["success"],
            failure=ours[i].get("failure"),
            geo=geo,
            detour=geo / max(euclid, 1e-9),
            turns=turns,
            corridor=2 * narrow,  # free width at the narrowest 10% of the path (clearance x 2, minus robot)
            visible=bool(visible),
            progress=1 - ours[i]["final_geo"] / max(geo, 1e-9),
        )
    )


def rate(rs, key):
    return 100 * np.mean([r[key] for r in rs]) if rs else float("nan")


def bins(feature, edges, fmt):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        rs = [r for r in rows if lo <= r[feature] < hi]
        out.append((fmt(lo, hi), len(rs), rate(rs, "ours"), rate(rs, "base")))
    return out


tables = {
    "path length (units)": bins(
        "geo", [0, 5, 8, 12, 16, 99], lambda a, b: f"{a:g}-{b:g}" if b < 99 else f"{a:g}+"
    ),
    "detour factor (path / straight line)": bins(
        "detour", [1, 1.3, 1.8, 2.5, 4, 99], lambda a, b: f"{a:g}-{b:g}" if b < 99 else f"{a:g}+"
    ),
    "turns along the path": bins(
        "turns", [0, 2, 4, 6, 9, 99], lambda a, b: f"{a}-{b - 1}" if b < 99 else f"{a}+"
    ),
    "corridor clearance (narrowest 10%)": bins(
        "corridor", [0, 0.7, 0.8, 0.9, 1.0, 9], lambda a, b: f"{a:g}-{b:g}" if b < 9 else f"{a:g}+"
    ),
}
vis = [
    ("B visible from A", [r for r in rows if r["visible"]]),
    ("B hidden from A", [r for r in rows if not r["visible"]]),
]

lines = [f"# {args.category}: {Path(args.eval_dir).parent.name} vs baseline ({len(rows)} test maps)", ""]
lines.append(f"Overall: ours {rate(rows, 'ours'):.1f}% vs baseline {rate(rows, 'base'):.1f}%")
both = sum(r["ours"] and r["base"] for r in rows)
only_b = sum(r["base"] and not r["ours"] for r in rows)
only_o = sum(r["ours"] and not r["base"] for r in rows)
neither = sum(not r["ours"] and not r["base"] for r in rows)
lines.append(f"Both solve {both} | only baseline {only_b} | only ours {only_o} | neither {neither}")
lines.append("")
for title, data in tables.items():
    lines += [f"## {title}", "", "| bin | maps | ours | baseline |", "|---|---|---|---|"]
    lines += [f"| {b} | {n} | {o:.0f}% | {s:.0f}% |" for b, n, o, s in data]
    lines.append("")
lines += ["## line of sight", "", "| | maps | ours | baseline |", "|---|---|---|---|"]
lines += [f"| {t} | {len(rs)} | {rate(rs, 'ours'):.0f}% | {rate(rs, 'base'):.0f}% |" for t, rs in vis]
fails = [r for r in rows if not r["ours"]]
lines += ["", "## our failures", ""]
for kind in ("looping", "stuck", "progressing"):
    rs = [r for r in fails if r["failure"] == kind]
    if rs:
        lines.append(
            f"- {kind}: {len(rs)} ({100 * len(rs) / len(rows):.1f}% of maps), "
            f"median progress before failing {100 * np.median([r['progress'] for r in rs]):.0f}% of the path"
        )
prog = np.array([r["progress"] for r in fails])
lines.append(
    f"- where it gives out: <25% of the way {100 * np.mean(prog < 0.25):.0f}%, 25-75% {100 * np.mean((prog >= 0.25) & (prog < 0.75)):.0f}%, "
    f">75% {100 * np.mean(prog >= 0.75):.0f}% (of failures)"
)
out = Path(args.eval_dir)
(out / f"analysis_{args.category}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# figure: success vs path length and turns, ours vs baseline
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (title, data) in zip(axes, list(tables.items())[:2] + [list(tables.items())[2]], strict=True):
    x = np.arange(len(data))
    ax.bar(x - 0.2, [d[2] for d in data], 0.4, label="learned")
    ax.bar(x + 0.2, [d[3] for d in data], 0.4, label="baseline")
    ax.set_xticks(x, [f"{d[0]}\n(n={d[1]})" for d in data], fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("success %")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / f"analysis_{args.category}.png", dpi=90)
