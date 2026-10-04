# ruff: noqa: E501
"""Build artifacts/release/versions/: every demoable Wheely, with the story of how it got better.

uv run python scripts/make_versions.py [--skip-eval] [--skip-gifs]

Each version gets <id>/weights.json (with the exact sim_params it was trained in), <id>/eval/ (test set, 1,000 maps
per category) and, unless --skip-gifs, a matplotlib GIF on the "Rooms" preset (research figure; the write-up clips
are recorded from the web demo). Writes versions.json (manifest for the web demo: live flag, labels, arch,
sim_params, test-set success) and versions_presets.png (every version on every preset). The heuristic has no weights: the web port reimplements
mazebot/baselines.py:ReactivePolicy.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from mazebot import mapgen
from mazebot.env import MazeEnv
from mazebot.evaluate import make_policy, run_episode
from mazebot.render import ACCENT, draw_map, draw_rays

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/release/versions"
RUNS = ROOT / "artifacts/runs"

# Chronological story. "live": playable in the web demo; the rest appear as clips/numbers in the write-up.
# "clip": preset the write-up clip is recorded on (from the web demo, M4). "sim_fix" patches sim_params that older
# files did not record.
VERSIONS = [
    dict(
        id="heuristic",
        name="By-the-Book",
        source=None,
        stage="baseline",
        live=True,
        clip="Rooms",
        summary="No learning, just rules written by hand: steer to the goal, turn towards open space when blocked, "
        "keep a close wall on the goal side.",
        lesson="A strong, simple baseline. Every learned Wheely has to beat it.",
    ),
    dict(
        id="v0-moonwalker",
        name="Moonwalker",
        source="v0_blind_reverse/val/step_005000160.json",
        sim_fix={"reverse_max": 1.0},
        stage="failure",
        live=False,
        clip="Open field",
        summary="8 inputs, 8-6-6-2, reversing allowed. Learned to drive backwards toward goals behind it — blind, "
        "because its rays only look forward.",
        lesson="RL takes the reward literally: reversing at the goal is 'efficient' until a wall it can't see is "
        "in the way. Training noise hid the problem; the noise-free policy got worse with more training.",
    ),
    dict(
        id="v2-scaredy",
        name="Scaredy-Wheely",
        source="v2_probe/val/step_003000096.json",
        stage="failure",
        live=False,
        clip="Slalom",
        summary="No reversing, but a big penalty for touching walls (0.2/step).",
        lesson="When touching a wall costs more than standing still, 'stop when unsure' becomes the best strategy.",
    ),
    dict(
        id="v3-rookie",
        name="Rookie",
        source="mix_6x6_s0/weights.json",
        stage="milestone",
        live=True,
        clip="Small maze",
        summary="First Wheely that learned to drive: 8 inputs, 8-6-6-2 (110 numbers), no reversing.",
        lesson="Level with By-the-Book on open and cluttered layouts; much worse in mazes.",
    ),
    dict(
        id="v4-owl-eyes",
        name="Owl Eyes",
        source="mix_360_12x12_s0/weights.json",
        stage="milestone",
        live=True,
        clip="Rooms",
        summary="7 rays all the way around (every ~51°), 10-12-12-2.",
        lesson="Seeing behind helps it notice it drove into a pocket: traps 18% → 64%, mazes 44% → 56%.",
    ),
    dict(
        id="v5-memory",
        name="One-step memory",
        source="mix_fwd5mem_12x12_s0/weights.json",
        stage="experiment",
        live=False,
        clip=None,
        summary="5 forward rays + the previous wheel commands as inputs, 10-12-12-2.",
        lesson="Remembering its last move was the biggest single gain; the final Wheely builds on it.",
    ),
    dict(
        id="final",
        name="Wheely",
        source="mix_fwd5mem_12x12_pcrit_s0/weights.json",
        stage="final",
        live=True,
        clip="Slalom",
        summary="Remembers its last move: memory inputs, 10-12-12-4-2 (350 numbers), trained with a privileged critic.",
        lesson="Matches or beats By-the-Book in 6 of 7 test categories. Still loses in winding mazes: it won't "
        "commit to a long detour away from B.",
    ),
    dict(
        id="gru",
        name="Recurrent memory (GRU)",
        source="mix_fwd5mem_gru8x4_s0/weights.json",
        stage="experiment",
        live=False,
        clip=None,
        summary="GRU with 8 memory neurons → 4 → 2, trained on the GPU.",
        lesson="Longer memory alone didn't fix mazes (55%).",
    ),
]


def build_files():
    OUT.mkdir(parents=True, exist_ok=True)
    for v in VERSIONS:
        d = OUT / v["id"]
        d.mkdir(exist_ok=True)
        if v["source"]:
            w = json.loads((RUNS / v["source"]).read_text(encoding="utf-8"))
            w["sim_params"].update(v.get("sim_fix", {}))
            w.setdefault("meta", {})["source"] = v["source"]
            (d / "weights.json").write_text(json.dumps(w), encoding="utf-8", newline="\n")


def spec(v):
    return str(OUT / v["id"] / "weights.json") if v["source"] else "reactive"


def evaluate(v):
    out = OUT / v["id"] / "eval"
    if v["source"] is None:  # baseline already evaluated
        shutil.copytree(ROOT / "artifacts/eval/reactive", out, dirs_exist_ok=True)
        return
    src = (RUNS / v["source"]).parent / "eval"
    if v["source"].endswith("weights.json") and (src / "report.json").exists() and not v.get("sim_fix"):
        shutil.copytree(src, out, dirs_exist_ok=True)  # same weights, same sim: reuse the test-set run
        return
    py = ROOT / ".venv/Scripts/python.exe"  # exported GRU inference is plain NumPy
    subprocess.run(
        [
            str(py),
            "-m",
            "mazebot.evaluate",
            "--policy",
            spec(v),
            "--n",
            "1000",
            "--workers",
            "12",
            "--out",
            str(out),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def gif(v, preset, path, every=2):
    pol = make_policy(spec(v))
    m = mapgen.attach_field(mapgen.preset_map(preset))
    env = MazeEnv(maps=[m], sim=getattr(pol, "sim", None))
    obs, _ = env.reset(options={"map": m})
    if hasattr(pol, "reset"):
        pol.reset()
    fig = plt.figure(figsize=(m.width / 2.2, m.height / 2.2), dpi=60)
    frames, xs, ys, k = [], [env.x], [env.y], 0
    while True:
        obs, _, term, trunc, _ = env.step(pol(obs))
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


def grid(versions):
    presets = mapgen.load_presets()
    fig, axes = plt.subplots(len(versions), len(presets), figsize=(len(presets) * 2.8, len(versions) * 1.95))
    for i, v in enumerate(versions):
        for j, p in enumerate(presets):
            m = mapgen.attach_field(mapgen.preset_map(p))
            r = run_episode(make_policy(spec(v)), m, record=True)
            ax = axes[i, j]
            res = f"{r['time_s']:.0f}s" if r["success"] else r["failure"]
            draw_map(ax, m, show_field=False, title=f"{p['name']}: {res}" if i == 0 else res)
            ax.plot(*r["traj"], color=ACCENT, lw=1.1)
            if j == 0:
                ax.set_ylabel(v["name"].split(":")[0][:22], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "versions_presets.png", dpi=70)
    plt.close(fig)


def manifest():
    cats = mapgen.CATEGORIES[:7]
    out = []
    for v in VERSIONS:
        rep = json.loads((OUT / v["id"] / "eval/report.json").read_text(encoding="utf-8"))["categories"]
        entry = {k: v[k] for k in ("id", "name", "stage", "live", "clip", "summary", "lesson")}
        entry["kind"] = "heuristic" if v["source"] is None else ("gru" if v["id"] == "gru" else "mlp")
        entry["weights"] = None if v["source"] is None else f"{v['id']}/weights.json"
        if v["source"]:
            w = json.loads((OUT / v["id"] / "weights.json").read_text(encoding="utf-8"))
            entry["arch"] = w["arch"]
            entry["sim_params"] = w["sim_params"]
        entry["test_success"] = {c: round(rep[c]["success"], 4) for c in cats}
        out.append(entry)
    (OUT / "versions.json").write_text(
        json.dumps({"versions": out}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = ["| version | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for e in out:
        lines.append(
            f"| {e['name']} | " + " | ".join(f"{100 * e['test_success'][c]:.1f}" for c in cats) + " |"
        )
    (OUT / "versions_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-gifs", action="store_true")
    args = ap.parse_args()
    build_files()
    for v in VERSIONS:
        if not args.skip_eval:
            evaluate(v)
            print("eval", v["id"], flush=True)
    if not args.skip_gifs:
        rooms = next(p for p in mapgen.load_presets() if p["name"] == "Rooms")
        for v in VERSIONS:
            gif(v, rooms, OUT / v["id"] / "rooms.gif")
            print("gif", v["id"], flush=True)
        grid(VERSIONS)
    manifest()
    sys.exit(0)
