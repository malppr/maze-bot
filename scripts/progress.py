"""Visualise how a run's policy evolves: validation curves + trajectories of every checkpoint on fixed maps.

uv run python scripts/progress.py artifacts/runs/<run> [--maps 5]
Writes <run>/progress/curves.png and <run>/progress/trajectories.png (rows = checkpoints, cols = maps).
Uses the validation checkpoints (val/step_*.json) the trainer saves every N steps.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mazebot import mapgen
from mazebot.evaluate import VAL_SEED, eval_map, make_policy, run_episode
from mazebot.render import ACCENT, draw_map

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--maps", type=int, default=5, help="number of fixed validation maps (one per category)")
ap.add_argument("--max-rows", type=int, default=8)
args = ap.parse_args()

run = Path(args.run)
out = run / "progress"
out.mkdir(exist_ok=True)

# ---- 1. validation curves per category, with curriculum stage changes
recs = [json.loads(line) for line in open(run / "validation.jsonl", encoding="utf-8")]
steps = [r["step"] / 1e6 for r in recs]
fig, ax = plt.subplots(figsize=(8, 4.2))
for cat in mapgen.TRAIN_CATEGORIES:
    ax.plot(steps, [r[cat] for r in recs], marker="o", ms=3, lw=1.2, label=cat)
ax.plot(steps, [r["score"] for r in recs], color="black", lw=2.2, label="score (mean)")
prev = recs[0]["stage"]
for r in recs:
    if r["stage"] != prev:
        ax.axvline(r["step"] / 1e6, color="grey", ls=":", lw=1)
        ax.text(r["step"] / 1e6, 0.02, f" stage {r['stage']}", color="grey", fontsize=7, rotation=90)
        prev = r["stage"]
best = max(recs, key=lambda r: r["score"])
ax.scatter([best["step"] / 1e6], [best["score"]], s=120, facecolor="none", edgecolor="black", zorder=5)
ax.set_xlabel("environment steps (millions)")
ax.set_ylabel("validation success rate")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=3, loc="lower right")
ax.set_title(f"{run.name}: validation success per category (circle = best checkpoint)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "curves.png", dpi=100)
plt.close(fig)

# ---- 2. trajectory grid: same validation maps, every checkpoint
ckpts = sorted((run / "val").glob("step_*.json"))
if len(ckpts) > args.max_rows:  # evenly subsample, always keep first and last
    idx = [round(i * (len(ckpts) - 1) / (args.max_rows - 1)) for i in range(args.max_rows)]
    ckpts = [ckpts[i] for i in idx]
cats = list(mapgen.TRAIN_CATEGORIES)[: args.maps]
maps = [eval_map(c, 3, seed=VAL_SEED) for c in cats]
fig, axes = plt.subplots(len(ckpts), len(maps), figsize=(len(maps) * 2.6, len(ckpts) * 2.3), squeeze=False)
for i, ck in enumerate(ckpts):
    pol = make_policy(str(ck))
    for j, m in enumerate(maps):
        r = run_episode(pol, m, record=True)
        ax = axes[i, j]
        res = "ok" if r["success"] else r["failure"]
        draw_map(ax, m, show_free=False, title=f"{cats[j]} — {res}, {r['time_s']:.0f}s")
        xs, ys = r["traj"]
        ax.plot(xs, ys, color=ACCENT, lw=1.0)
        if j == 0:
            ax.set_ylabel(f"{int(ck.stem.split('_')[1]) / 1e6:.0f}M steps", fontsize=9)
fig.suptitle(f"{run.name}: one validation map per category (columns) at each checkpoint (rows)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "trajectories.png", dpi=75)
plt.close(fig)
print(f"wrote {out / 'curves.png'} and {out / 'trajectories.png'} ({len(ckpts)} checkpoints)")
