"""Env throughput (single process): random actions, training map mix, resets included."""

import time

import numpy as np

from mazebot import mapgen
from mazebot.env import MazeEnv

for cats in [("open",), ("mazes",), ("scribbles",), mapgen.TRAIN_CATEGORIES]:
    env = MazeEnv(categories=cats)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    steps = resets = 0
    reset_t = 0.0
    t0 = time.perf_counter()
    while steps < 5000:
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        steps += 1
        if term or trunc:
            t1 = time.perf_counter()
            env.reset()
            reset_t += time.perf_counter() - t1
            resets += 1
    dt = time.perf_counter() - t0
    print(
        f"{'+'.join(cats):40s} {steps / dt:7.0f} steps/s  "
        f"(step only {steps / (dt - reset_t):7.0f}/s, {resets} resets, "
        f"{1000 * reset_t / max(resets, 1):.0f} ms/reset, last map {len(env.caps)} caps)"
    )
