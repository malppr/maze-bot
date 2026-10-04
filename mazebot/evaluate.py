"""Held-out evaluation: N maps per category, seeds disjoint from training, per-category report.

Usage:
  uv run python -m mazebot.evaluate --policy artifacts/runs/<run>/weights.json --n 1000
  uv run python -m mazebot.evaluate --policy reactive
  uv run python -m mazebot.evaluate --policy random

Training maps come from Gymnasium's seeded `np_random` streams; evaluation maps come from
`default_rng([EVAL_SEED, category, i])`, an independent SeedSequence, so the sets don't overlap.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import mapgen
from .env import MazeEnv

EVAL_SEED = 20261004  # test set: final reported numbers only
VAL_SEED = 77001  # validation set: checkpoint selection / early stopping during training
WINDOW_S = 10.0  # failure classification looks at the last 10 s of a failed episode


def eval_map(category: str, i: int, seed: int = EVAL_SEED) -> mapgen.Map:
    rng = np.random.default_rng([seed, mapgen.CATEGORIES.index(category), i])
    return mapgen.sample_map(rng, category)


def make_policy(spec: str, seed: int = 0):
    from .baselines import RandomPolicy, ReactivePolicy
    from .policy import load_policy

    if spec == "random":
        return RandomPolicy(seed)
    if spec == "reactive":
        return ReactivePolicy()
    if "@noise=" in spec:  # e.g. weights.json@noise=0.3: Gaussian action noise like during training
        path, sigma = spec.split("@noise=")
        mlp, rng, sigma = load_policy(path), np.random.default_rng(seed), float(sigma)

        def noisy(obs):
            return np.clip(mlp(obs) + rng.normal(0.0, sigma, 2), -1.0, 1.0)

        noisy.sim = mlp.sim
        return noisy
    return load_policy(spec)


def classify_failure(xs, ys, geos, contacts, window: int) -> str:
    """Why did a timed-out episode fail? Looks at the last `window` policy steps.

    - progressing: still getting closer to B (along the path) — just too slow
    - stuck: barely moving (pinned against a wall, or spinning in place)
    - looping: moving but not getting closer (circling, oscillating, wall-following forever)
    """
    w = min(window, len(xs) - 1)
    progress = geos[-w - 1] - geos[-1]
    seg = np.hypot(np.diff(xs[-w - 1 :]), np.diff(ys[-w - 1 :]))
    travelled = float(seg.sum())
    if progress > 1.0:
        return "progressing"
    if travelled < 1.0:
        return "stuck"
    return "looping"


def run_episode(policy, m: mapgen.Map, record: bool = False) -> dict:
    env = MazeEnv(maps=[m], sim=getattr(policy, "sim", None))  # the policy's own sensor layout
    obs, info = env.reset(seed=0, options={"map": m})
    if hasattr(policy, "reset"):  # recurrent policies start each episode with empty memory
        policy.reset()
    xs, ys, geos, contacts = [env.x], [env.y], [env.geo], [False]
    acts = []
    reversing = 0
    while True:
        a = policy(obs)
        obs, _, term, trunc, info = env.step(a)
        xs.append(env.x)
        ys.append(env.y)
        geos.append(env.geo)
        contacts.append(info["contact"])
        if record:
            acts.append(np.asarray(a, dtype=np.float64))
        if a[0] + a[1] < -0.2:  # commanded backwards
            reversing += 1
        if term or trunc:
            break
    policy_dt = env.sim.dt * env.sim.frame_skip
    res = {
        "category": m.category,
        "success": bool(info["success"]),
        "steps": info["steps"],
        "time_s": info["steps"] * policy_dt,
        "contacts": info["contacts"],
        "geo0": env.geo0,
        "path_len": env.path_len,
        # SPL-style efficiency: geodesic optimum / distance actually driven (0 on failure)
        "spl": (env.geo0 / max(env.path_len, env.geo0)) if info["success"] else 0.0,
        "final_geo": env.geo,
        "reverse_frac": reversing / max(info["steps"], 1),
    }
    if not res["success"]:
        res["failure"] = classify_failure(
            np.array(xs), np.array(ys), np.array(geos), np.array(contacts), int(WINDOW_S / policy_dt)
        )
    if record:
        res["traj"] = (np.array(xs), np.array(ys))
    return res


def _worker(args):
    spec, category, indices, seed = args
    out = []
    for i in indices:
        policy = make_policy(spec, seed=i)
        r = run_episode(policy, eval_map(category, i, seed))
        r["index"] = i
        out.append(r)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    succ = [r for r in results if r["success"]]
    k = len(succ)
    lo, hi = wilson(k, n)
    fails = Counter(r["failure"] for r in results if not r["success"])
    return {
        "n": n,
        "success": k / n if n else 0.0,
        "ci95": [lo, hi],
        "spl": float(np.mean([r["spl"] for r in results])) if n else 0.0,
        "time_to_goal_s": float(np.mean([r["time_s"] for r in succ])) if succ else None,
        "contacts_per_ep": float(np.mean([r["contacts"] for r in results])) if n else 0.0,
        "reverse_frac": float(np.mean([r["reverse_frac"] for r in results])) if n else 0.0,
        "failures": {f: fails.get(f, 0) / n for f in ("stuck", "looping", "progressing")},
    }


def evaluate(spec: str, n: int, categories=mapgen.CATEGORIES, workers: int = 14, seed: int = EVAL_SEED):
    jobs = []
    chunk = max(1, n // (workers * 2))
    for cat in categories:
        for s in range(0, n, chunk):
            jobs.append((spec, cat, list(range(s, min(n, s + chunk))), seed))
    results: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_worker, jobs):
            results.extend(part)
    report = {cat: summarize([r for r in results if r["category"] == cat]) for cat in categories}
    return {"policy": spec, "n_per_category": n, "eval_seed": seed, "categories": report}, results


def to_markdown(report: dict) -> str:
    lines = [
        f"Policy: `{report['policy']}` — {report['n_per_category']} held-out maps per category",
        "",
        "| category | success (95% CI) | SPL | time to goal | contacts/ep | stuck | looping | progressing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cat, s in report["categories"].items():
        t = f"{s['time_to_goal_s']:.1f} s" if s["time_to_goal_s"] is not None else "—"
        f = s["failures"]
        lines.append(
            f"| {cat} | {100 * s['success']:.1f}% ({100 * s['ci95'][0]:.1f}–{100 * s['ci95'][1]:.1f}) "
            f"| {s['spl']:.2f} | {t} | {s['contacts_per_ep']:.1f} "
            f"| {100 * f['stuck']:.1f}% | {100 * f['looping']:.1f}% | {100 * f['progressing']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="weights.json path, 'reactive' or 'random'")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--categories", default=",".join(mapgen.CATEGORIES))
    ap.add_argument("--out", default=None, help="output dir (default: <weights dir>/eval)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    report, results = evaluate(args.policy, args.n, args.categories.split(","), args.workers)
    report["seconds"] = time.perf_counter() - t0
    if args.out:
        out = Path(args.out)
    elif args.policy.endswith(".json"):
        out = Path(args.policy).parent / "eval"
    else:
        out = Path("artifacts/eval") / args.policy
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(to_markdown(report))
    print(f"[{report['seconds']:.0f} s] -> {out}")


if __name__ == "__main__":
    main()
