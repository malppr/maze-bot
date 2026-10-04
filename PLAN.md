# Wheely's maze (repo: maze-bot) — Plan

Wheely, the site's two-wheeled robot mascot, learns with reinforcement learning to drive from A to B
using five distance rays and the direction to the goal. Visitors pick a preset layout or **freehand-draw**
walls and watch it navigate, with every neuron of its network visible.

Consumed by the portfolio site — see `D:\Proj\malppr.github.io\PLAN.md` §8 (integration contract) and §14 (decisions).
The demo is titled **"Wheely's maze"**; the repo/package keeps the name `maze-bot`.

---

## 1. What the visitor experiences (Playground → "Wheely's maze")

- **Presets** mode: a handful of hand-designed layouts (open field, posts, corridor, zig-zag, small maze, a trap)
  with A and B; a "shuffle A/B" button.
- **Draw** mode: freehand walls with mouse/touch (**no grid snapping**), eraser, clear; drag A and B anywhere.
  Starting a preset and then drawing on top of it is allowed.
- Rays drawn as dashed lines (redder = closer), faint goal-direction line, Wheely as the top-down sprite.
- **Network panel:** 8 → 6 → 6 → 2, **every neuron drawn** with its live value; nodes glow by activation,
  edges colored by sign, thickness by |w|; hover/tap a node to highlight its connections. Inputs and outputs labelled
  (ray 1–5, goal sin/cos, goal distance → left wheel, right wheel).
- If B is unreachable: **warning "No path to B — Wheely will get lost"**, but Wheely still runs.
  A "lost" state after a timeout or no progress, with a reset button. Getting lost is part of the show.
- Speed (0.5× / 1× / 4×), pause, step, reset.
- Reduced motion: starts paused; runs only on an explicit Play.

No roaming mascot elsewhere on the site (decided 2026-10-04) — the Playground is Wheely's only home.

## 2. Robot & world

- **Coordinates:** continuous 2D, screen convention — x right, **y down**. Heading θ = 0 points along +x;
  positive θ turns clockwise on screen. One convention in Python and TS, so parity tests compare raw numbers.
- **Arena:** rectangle whose border is always a wall. Size is **randomized in training** (each side 6–12 units)
  so the same policy works in the desktop (landscape, 10 × 6.25) and phone (portrait) arenas.
- **Robot:** circle collider, radius `r = 0.3`. Differential drive with wheel commands `uL, uR ∈ [−1, 1]`:
  `v = vmax · (uL + uR)/2`, `ω = vmax · (uL − uR)/b` (y-down: left wheel faster ⇒ turn clockwise/right).
  Physics at `dt = 1/30 s`, policy at 15 Hz (action held for 2 physics steps).
- **Walls:** everything is a **capsule** (segment + radius): freehand strokes, preset obstacles, maze walls, the arena border.
  One representation everywhere; rendered as round-capped strokes, which also suits the hand-drawn look.
- **Collision:** circle vs capsule. Sub-step so no step moves more than `r/2` (no tunneling through thin strokes);
  on contact, push out along the contact normal ⇒ Wheely slides along walls. Deterministic iteration order (for parity).
- **Rays:** 5 rays at −60°, −30°, 0°, 30°, 60° from the heading, max range `R = 3`. Analytic ray–capsule intersection,
  brute force over all capsules (strokes are simplified, so N stays in the low hundreds; a spatial hash only if profiling asks for it).

### Observation (8 inputs)

| # | Input | Range |
|---|---|---|
| 0–4 | ray distance / R | [0, 1] |
| 5–6 | sin, cos of goal bearing relative to heading | [−1, 1] |
| 7 | goal distance / 10, clipped | [0, 1] |

No observation normalization wrapper (inputs are already scaled), so exported weights need no extra statistics.

### Action (2 outputs)

Two outputs → `uL, uR`, **linear layer clipped to [−1, 1]** (the standard SB3 Box-action setup; deterministic action = clip(mean)).
Sign gives direction, so spin-in-place and reverse are free.

### Network

MLP **8 → 6 → 6 → 2**, tanh hidden layers, 104 parameters — small enough to draw every weight and neuron.
Hidden width stays a config value: if 6-6 clearly can't learn freehand maps, compare with 8-8 and a 64-64 upper bound in M2
and decide together (readability of the network panel matters).

## 3. Freehand walls without grid snapping

- **Stroke processing (browser):** pointer samples → Ramer–Douglas–Peucker simplification → chain of capsules with the brush radius.
  Cap total capsule count (performance on phones).
- **Path check (warning only):** rasterize capsules to an occupancy grid (cell = r/2), **inflate by the robot radius**,
  BFS from A. B unreachable ⇒ warning, still run. Gaps narrower than Wheely correctly count as blocked.
  Same algorithm as the training distance field (§4), ported to TS.
- **Training covers freehand-looking walls** so drawn maps are in distribution. Map generator categories:
  1. `open` — empty arena (sanity).
  2. `obstacles` — random posts and bars.
  3. `scribbles` — smoothed random-walk polylines, varying thickness, some closed loops.
  4. `mazes` — recursive-backtracker mazes converted to capsules, with wobble so they look hand-drawn.
  5. `mixed` — combinations of the above.
  6. `traps` — U-shapes and pockets facing the start (evaluation only at first; known hard case for a memoryless policy).
  7. `presets` — the demo's own preset layouts (shared JSON read by Python and TS), random A/B.
  Training A/B placement: rejection-sample until B is reachable and the geodesic distance is above a minimum.
- **Expected behaviour (stated up front):** with no memory the policy will learn something like a **Bug algorithm** —
  head to goal, follow a wall when blocked, leave it when the goal is clear. It will fail on some layouts
  (wall-following around islands, U-traps). That's a feature for the write-up and for "draw a trap" moments in the demo.

## 4. Training

- **Env:** Gymnasium API, pure NumPy, float64, seeded; vectorized with SB3 `SubprocVecEnv` (16 logical cores available).
- **Algorithm:** PPO via **Stable-Baselines3** (CPU PyTorch). Actor matches the export architecture exactly
  (`net_arch=dict(pi=[6, 6], vf=[64, 64])`, tanh). The value net can be big — it is not exported.
- **Reward (per policy step):**
  - `+ k_p · (d_geo(prev) − d_geo(now))` — progress in **geodesic** distance (distance field from B on the inflated grid,
    computed once per map, bilinearly interpolated). Euclidean progress would trap it in dead ends near B.
  - `− k_c` per step in wall contact
  - `− k_t` small time penalty
  - `+ R_goal` on reaching B (episode ends); timeout (scaled by the geodesic A→B distance) ends with no bonus.
- **Curriculum:** open → obstacles → scribbles → mazes → mixed; advance when rolling success > threshold; keep a share of earlier stages.
- **Evaluation:** fixed seeded held-out set, **1,000 maps per category** (seeds disjoint from training). Report per category:
  success rate (with 95% CI), mean time-to-goal, collisions/episode, and a failure taxonomy (stuck against a wall,
  looping/oscillating, timeout while progressing). Learning curves via TensorBoard; rollout GIFs of successes and failures.
- **Baselines for the write-up:** random policy; hand-coded Bug-style controller on the same 8 inputs;
  wider nets (8-8, 64-64) as an "is the tiny net the bottleneck?" check. (Optional later: CMA-ES on the same net.)
- **Training-distribution ablation:** mazes-only vs full mix, both evaluated on every category — what does shape
  variety cost on mazes and buy on scribbles/presets/traps? Large cost on mazes ⇒ evidence for 8-8.
- **Compute:** CPU, minutes to ~1 h per run.

## 5. Export & browser runtime

- `weights.json`: `{ version, arch: [8,6,6,2], hidden_activation: "tanh", output: "clip", obs_spec, sim_params, W, b }` — a few KB.
- **TypeScript sim** (`web/src/sim/`) is a faithful port of the Python env (kinematics, capsules, rays, collision, observation, grid BFS).
- **Parity tests** (CI): fixtures generated by Python, replayed in TS —
  single-step parity from many random states (≤ 1e-9), short seeded rollouts (≤ 1e-6 over 300 steps),
  ray/observation parity, and policy-forward parity (≤ 1e-6). Long rollouts can legitimately diverge after
  a grazing contact (1-ulp `sin` differences), so parity is checked per step, not on 10-minute trajectories.
- Inference = three tiny matmuls in plain TS; no ONNX.

## 6. Repo layout

```
maze-bot/
  pyproject.toml            # uv-managed; deps: numpy, gymnasium; groups: dev (pytest, ruff, matplotlib), train (torch cpu, sb3, tensorboard)
  mazebot/
    geometry.py  sim.py  env.py  distance_field.py  mapgen.py  render.py
    train.py  evaluate.py  export.py  baselines.py
  presets/                  # demo preset layouts (JSON), shared by Python and TS
  scripts/                  # make_presets.py, render_samples.py, bench_env.py
  tests/                    # geometry, rays, collision, kinematics, distance field, mapgen, env/reward
  configs/                  # training configs (yaml)
  artifacts/                # exported weights (per release), eval reports, sample renders
  package.json              # npm entry at repo root (npm git deps need it there): exports "maze-bot/web"
  web/
    src/sim/                # TS port
    src/demo/               # mountMazeDemo: canvas, presets, drawing, path-check warning, network panel, controls
    src/assets/             # copy of the site's mascot-top.svg (package is self-contained)
    tests/                  # parity + unit tests (vitest)
    dev/                    # standalone Vite dev page
  .github/workflows/ci.yml  # ruff + pytest (+ vitest + parity from M3)
  PLAN.md  CLAUDE.md  README.md
```

## 7. Integration contract (site side)

```ts
import { mountMazeDemo } from "maze-bot/web";

mountMazeDemo(el: HTMLElement, opts?: {
  mode?: "obstacles" | "draw";   // "obstacles" = presets mode
  showNetwork?: boolean;          // neuron panel
}): { destroy(): void };
```

- Theme: reads `--fg`, `--bg`, `--surface`, `--border`, `--muted`, `--accent` from the container (re-read on theme change). No theme props.
- The demo owns its layout inside `el` (arena, controls, network panel; panel moves below the arena on phones).
  The site's fixed-aspect `.stage` box will need loosening when the real demo is wired in.
- Pauses when off-screen or tab hidden; respects reduced motion.
- `mountMascot` is **dropped** (no roaming mascot in the site hero).
- Site wiring: `src/demos/maze.ts` re-exports from `maze-bot/web`, `IS_PREVIEW = false`, dependency
  `"maze-bot": "github:malppr/maze-bot#v0.1.0"`.

## 8. Milestones

Each milestone ends with a check run by Claude (tests, metrics, screenshots) and a short report to Bryan before moving on.

| # | Deliverable | Done when |
|---|---|---|
| M0 | `uv` project (uv installs Python 3.12), git repo, CI skeleton, CLAUDE.md, README stub | `uv run pytest` + `uv run ruff check` green locally; CI file ready (runs once Bryan pushes) |
| M1 | Geometry (capsules, rays, collision), kinematics, env, distance field, map generator (all categories + presets), matplotlib renderer | Unit tests + `check_env` pass; rendered sample sheets of every category reviewed; env steps/s measured |
| M2 | PPO training + curriculum + held-out evaluation + baselines | Per-category held-out success rates, failure taxonomy, rollout GIFs; hidden size decided with Bryan |
| M3 | `weights.json` export + TS sim port + parity tests | Parity tests pass locally and in CI |
| M4 | `mountMazeDemo`: canvas, presets, draw mode, path-check warning, network panel, controls | Works in the dev page; screenshots desktop + 390 px, light + dark, reviewed |
| M5 | Package release `v0.1.0`; site wiring (`maze.ts`, `IS_PREVIEW = false`, stage layout) | Site `npm run check:all` passes; Bryan reviews `npm run dev` + `npm run preview`; tag ready to push |
| M6 | Write-up (method, reward design, results, failure modes, GIFs) → README + site project page "Wheely's maze" (tags robot-learning + ai) | Bryan review |

## 9. Risks

- **Freehand OOD:** visitors draw shapes unlike training maps → scribble generator + thickness randomization; accept and showcase failures.
- **6-6 too small** for robust navigation → measure in M2 against 8-8 / 64-64; options: wider hidden layers or more rays (changes the 8-input design — discuss first).
- **Sim mismatch Python↔TS** → parity tests are mandatory, not optional.
- **Env speed in pure NumPy** → measure in M1; vectorize ray casting over capsules; numba only if needed.
- **Mobile performance** with many strokes → RDP simplification; cap stroke count.

## 10. Open decisions

- [x] Mascot top-down sprite: use the site's `src/assets/mascot/mascot-top.svg` (faces −y ⇒ draw rotated by θ + 90°).
- [x] Name: Wheely; demo title "Wheely's maze"; no `mountMascot`.
- [ ] Final hidden size (6-6 vs 8-8) — after M2 results.
- [ ] Exact ray angles / range — tune in M2 if results call for it.
- [ ] License for the public repo (proposal: MIT for code; mascot art stays © Bryan Chew).
- [ ] "Train it live" (in-browser neuroevolution) — later, after M6.

## 11. Decisions log

- **2026-10-04 — Plan updated** for: Wheely naming, no roaming mascot (`mountMascot` dropped), freehand walls without snapping
  plus presets, "no path" warning that still lets Wheely run, 8 → 6 → 6 → 2 with every neuron visualised.
- **2026-10-04 — Train on random shapes, not strict mazes** (drawing is the main interaction; failures should come from
  missing memory, not OOD shapes). Mazes-only vs mixed ablation in M2. Varying arena size, clipped linear outputs,
  per-step parity accepted.
