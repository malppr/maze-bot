"""Write parity fixtures for the TypeScript port (web/test/fixtures/*.json).

uv run python scripts/make_parity_fixtures.py

The Python sim is the reference; web/test/*.test.ts replays every case and compares:
geometry.json (physics steps, rays, push-out, clearance, RDP), grid.json (occupancy grid, geodesic field,
reachability), policy.json (forward passes of every live version), rollouts.json (closed-loop episodes of
every live version on the presets and a few seeded maps). JSON floats round-trip exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mazebot import distance_field as dfield
from mazebot import geometry, mapgen
from mazebot.baselines import ReactivePolicy
from mazebot.env import MazeEnv
from mazebot.policy import MLPPolicy
from mazebot.sim import SimParams, observe, physics_step

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web/test/fixtures"
VERSIONS = ROOT / "artifacts/release/versions"
LIVE = ["heuristic", "v3-rookie", "v4-owl-eyes", "final"]
SEED = 20261004
ROLLOUT_STEPS = 300


def load_version(vid: str):
    """(policy, SimParams) for a live version. The heuristic sees the default (5 forward rays) sim."""
    if vid == "heuristic":
        return ReactivePolicy(), SimParams()
    pol = MLPPolicy.load(VERSIONS / vid / "weights.json")
    return pol, pol.sim


def sample_maps(rng, n_per_cat: int):
    cats = ["obstacles", "scribbles", "mazes", "mixed", "traps"]
    return [mapgen.sample_map(rng, c) for c in cats for _ in range(n_per_cat)]


def map_json(m: mapgen.Map) -> dict:
    return {
        "name": m.name or m.category,
        "width": m.width,
        "height": m.height,
        "walls": m.walls.tolist(),
        "start": list(m.start),
        "goal": list(m.goal),
    }


def geometry_cases(rng, maps):
    sims = {"fwd5": SimParams(), "rev": SimParams(reverse_max=1.0), "owl": load_version("v4-owl-eyes")[1]}
    steps, rays, pushes, obs = [], [], [], []
    for m in maps:
        caps = m.caps
        for _ in range(30):
            x = rng.uniform(0.0, m.width)
            y = rng.uniform(0.0, m.height)
            th = rng.uniform(-math.pi, math.pi)
            ul, ur = (float(v) for v in rng.uniform(-1.0, 1.0, 2))
            if rng.random() < 0.15:  # spin in place / full speed: exact edge cases
                ul, ur = 1.0, -1.0
            elif rng.random() < 0.15:
                ul, ur = 1.0, 1.0
            for key in ("fwd5", "rev"):
                x2, y2, th2, c = physics_step(x, y, th, ul, ur, caps, sims[key])
                steps.append(
                    {"map": m.idx, "sim": key, "in": [x, y, th, ul, ur], "out": [x2, y2, th2, bool(c)]}
                )
            for key in ("fwd5", "owl"):
                o, r = observe(x, y, th, *m.goal, caps, sims[key])
                rays.append(
                    {"map": m.idx, "sim": key, "in": [x, y, th], "rays": r.tolist(), "obs": o.tolist()}
                )
        for _ in range(10):  # points inside / touching walls
            k = int(rng.integers(len(caps)))
            ax, ay, bx, by, rad = caps[k]
            t = rng.uniform(0.0, 1.0)
            ang = rng.uniform(-math.pi, math.pi)
            dist = rng.uniform(0.0, rad + 0.35)
            px = ax + t * (bx - ax) + dist * math.cos(ang)
            py = ay + t * (by - ay) + dist * math.sin(ang)
            ox, oy, c = geometry.push_out(px, py, 0.3, caps)
            pushes.append(
                {
                    "map": m.idx,
                    "in": [px, py],
                    "out": [ox, oy, bool(c)],
                    "clear": float(geometry.clearance(px, py, caps)),
                }
            )
    for m in maps[:3]:  # sincos / bearing goal inputs (not shipped, but supported by the port)
        for mode in ("sincos", "bearing"):
            p = SimParams(goal_inputs=mode)
            for _ in range(10):
                x, y, th = rng.uniform(0, m.width), rng.uniform(0, m.height), rng.uniform(-math.pi, math.pi)
                o, _ = observe(x, y, th, *m.goal, m.caps, p)
                obs.append({"map": m.idx, "goal_inputs": mode, "in": [x, y, th], "obs": o.tolist()})
    strokes = []
    for _ in range(20):
        n = int(rng.integers(2, 60))
        pts = np.cumsum(rng.normal(0.0, 0.08, size=(n, 2)), axis=0) + rng.uniform(1, 5, 2)
        keep = geometry.rdp(pts, 0.03)
        strokes.append({"points": pts.tolist(), "eps": 0.03, "kept": keep.tolist()})
    return {"steps": steps, "rays": rays, "pushes": pushes, "goal_modes": obs, "rdp": strokes}


def grid_cases(maps):
    out = []
    for m in maps:
        g = dfield.build_grid(m.width, m.height, m.caps, 0.3)
        f = dfield.geodesic_field(g, *m.goal)
        out.append(
            {
                "map": m.idx,
                "nx": g.free.shape[1],
                "ny": g.free.shape[0],
                "free": g.free.astype(int).ravel().tolist(),
                "field": [d if math.isfinite(d) else None for d in f.ravel().tolist()],
                "geo_start": float(dfield.lookup(g, f, m.start[0], m.start[1])),
                "reachable": bool(dfield.reachable(g, m.start[0], m.start[1], *m.goal)),
            }
        )
    return out


def policy_cases(rng):
    out = []
    for vid in LIVE:
        pol, sim = load_version(vid)
        obs = rng.uniform(-1.0, 1.0, size=(60, sim.obs_dim)).astype(np.float32)
        obs[:10, : len(sim.ray_angles)] = 1.0  # nothing in view
        cases = []
        for o in obs:
            if vid == "heuristic":
                cases.append({"obs": o.tolist(), "action": pol(o).tolist()})
            else:
                a, acts = pol.forward(o, return_activations=True)
                cases.append(
                    {"obs": o.tolist(), "action": a.tolist(), "acts": [h.tolist() for h in acts[1:]]}
                )
        out.append({"version": vid, "cases": cases})
    return out


def rollout_cases(maps):
    out = []
    for vid in LIVE:
        pol, sim = load_version(vid)
        for m in maps:
            mapgen.attach_field(m)
            env = MazeEnv(maps=[m], sim=sim)
            o, _ = env.reset(options={"map": m})
            traj, success, truncated = [[env.x, env.y, env.th]], False, False
            limit = env.max_steps if m.name == "The trap" else ROLLOUT_STEPS  # one run to the "lost" timeout
            for _ in range(limit):
                o, _, success, truncated, _ = env.step(pol(o))
                traj.append([env.x, env.y, env.th])
                if success or truncated:
                    break
            out.append(
                {
                    "version": vid,
                    "map": m.idx,
                    "max_steps": env.max_steps,
                    "success": bool(success),
                    "truncated": bool(truncated),
                    "traj": traj,
                }
            )
    return out


def main():
    rng = np.random.default_rng(SEED)
    presets = [mapgen.preset_map(p) for p in mapgen.load_presets()]
    presets += [mapgen.preset_map(p, portrait=True) for p in mapgen.load_presets()[:2]]
    random_maps = sample_maps(rng, 2)
    maps = presets + random_maps
    for i, m in enumerate(maps):
        m.idx = i
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "maps.json": [map_json(m) for m in maps],
        "geometry.json": geometry_cases(rng, random_maps + presets[3:5]),
        "grid.json": grid_cases(presets[:6] + random_maps[::2]),
        "policy.json": policy_cases(rng),
        "rollouts.json": rollout_cases(presets + random_maps[1::3]),
    }
    for name, data in files.items():
        (OUT / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8", newline="\n")
        print(f"{name}: {(OUT / name).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
