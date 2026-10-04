"""Trajectory sheets (successes + failures per category) and a few GIFs for a trained policy.

uv run python scripts/rollouts.py artifacts/runs/<run>/weights.json [--gifs 4]
Reads <run>/eval/episodes.jsonl (from mazebot.evaluate) to pick examples; writes <run>/rollouts/.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from mazebot import mapgen
from mazebot.evaluate import eval_map, make_policy, run_episode
from mazebot.render import ACCENT, draw_map, draw_rays

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--per", type=int, default=4, help="successes and failures per category")
ap.add_argument("--gifs", type=int, default=4)
args = ap.parse_args()

run = Path(args.weights).parent
out = run / "rollouts"
out.mkdir(exist_ok=True)
eps = [json.loads(line) for line in open(run / "eval" / "episodes.jsonl", encoding="utf-8")]

for cat in mapgen.CATEGORIES:
    ce = sorted((e for e in eps if e["category"] == cat), key=lambda e: e["index"])
    picks = [e for e in ce if e["success"]][: args.per] + [e for e in ce if not e["success"]][: args.per]
    if not picks:
        continue
    fig, axes = plt.subplots(2, args.per, figsize=(args.per * 3.4, 6.4))
    for ax in axes.ravel():
        ax.axis("off")
    succ = [e for e in picks if e["success"]]
    fail = [e for e in picks if not e["success"]]
    for row, group in enumerate((succ, fail)):
        for col, e in enumerate(group):
            m = eval_map(cat, e["index"])
            r = run_episode(make_policy(args.weights), m, record=True)
            ax = axes[row, col]
            ax.axis("on")
            label = "success" if r["success"] else r["failure"]
            draw_map(ax, m, show_free=False, title=f"#{e['index']} {label}, {r['time_s']:.0f}s")
            xs, ys = r["traj"]
            ax.plot(xs, ys, color=ACCENT, lw=1.1, alpha=0.85)
    fig.suptitle(f"{cat}: top row successes, bottom row failures (first {args.per} of each)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{cat}.png", dpi=80)
    plt.close(fig)
    print("sheet", cat)


def gif(m, path, policy, every=2):
    from mazebot.env import MazeEnv

    env = MazeEnv(maps=[m])
    obs, _ = env.reset(options={"map": m})
    frames, xs, ys = [], [env.x], [env.y]
    fig = plt.figure(figsize=(m.width / 2.2, m.height / 2.2), dpi=60)
    k = 0
    while True:
        obs, _, term, trunc, info = env.step(policy(obs))
        xs.append(env.x)
        ys.append(env.y)
        if k % every == 0 or term or trunc:
            fig.clf()
            ax = fig.add_axes([0, 0, 1, 1])
            draw_map(ax, m, show_field=False)
            ax.plot(xs, ys, color=ACCENT, lw=1, alpha=0.6)
            draw_rays(ax, env.x, env.y, env.th, env.rays, env.sim.ray_angles)
            ax.add_patch(Circle((env.x, env.y), env.sim.radius, facecolor=ACCENT, zorder=7))
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        k += 1
        if term or trunc:
            break
    plt.close(fig)
    imageio.mimsave(path, frames + [frames[-1]] * 10, duration=1 / 15 * every, loop=0)
    return info["success"]


presets = mapgen.load_presets()
for p in presets[: args.gifs + 2]:
    m = mapgen.attach_field(mapgen.preset_map(p))
    ok = gif(m, out / f"preset_{p['name'].lower().replace(' ', '_')}.gif", make_policy(args.weights))
    print("gif", p["name"], "success" if ok else "FAIL")
