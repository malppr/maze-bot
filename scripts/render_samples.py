"""Render sample maps of every category -> artifacts/samples/<category>.png (M1 review)."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mazebot import mapgen
from mazebot.render import draw_map

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="artifacts/samples")
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for cat in mapgen.CATEGORIES:
    rng = np.random.default_rng([args.seed, mapgen.CATEGORIES.index(cat)])
    t0 = time.perf_counter()
    maps = [mapgen.sample_map(rng, cat) for _ in range(args.n)]
    ms = (time.perf_counter() - t0) / args.n * 1000
    cols = 4
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.0))
    for ax, m in zip(axes.ravel(), maps, strict=False):
        label = f"{m.width:.1f}x{m.height:.1f}, {len(m.walls)} caps, geo {m.geo(*m.start[:2]):.1f}"
        draw_map(ax, m, title=(m.name + " | " if m.name else "") + label)
    for ax in axes.ravel()[len(maps) :]:
        ax.axis("off")
    fig.suptitle(
        f"{cat}  (blue = geodesic distance from B, grey = blocked for robot centre, orange = unreachable)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / f"{cat}.png", dpi=90)
    plt.close(fig)
    print(f"{cat:10s} {ms:6.1f} ms/map")
