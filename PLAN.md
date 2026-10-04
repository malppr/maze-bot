# Wheely's maze (repo: maze-bot) — Plan

Wheely, the site's two-wheeled robot mascot, learns with reinforcement learning to drive from A to B
using five distance rays and the direction to the goal. Visitors pick a preset layout or **freehand-draw**
walls and watch it navigate, with every neuron of its network visible.

Consumed by the portfolio site — see `D:\Proj\malppr.github.io\PLAN.md` §8 (integration contract) and §14 (decisions).
The demo is titled **"Wheely's maze"**; the repo/package keeps the name `maze-bot`.

---

## 0. Status (2026-10-05)

- **M0, M1, M2 done.** ~27 training runs; every experiment and result in §11.
- **Final policy: `artifacts/release/`** — `weights.json` (actor 10 → 12 → 12 → 4 → 2, memory inputs; from run
  `mix_fwd5mem_12x12_pcrit_s0`), preset GIFs, figures, test report, and the **sim/obs spec the TS port must match**
  (`artifacts/release/README.md`). Test set: matches/beats the hand-coded baseline in 6 of 7 categories; mazes
  59.7% vs 68.6%. Reaches B on 5 of 6 presets; loops in "The trap" (intended demo moment).
- **Next: web wiring — M3 (TS port + parity), M4 (demo), M5 (package + site), M6 (write-up + project page).**
  See `HANDOFF.md`. Part 4 planning (PushT, robot-arm page, AI side project) comes after M6.

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

### Observation (10 inputs — current leading design; originally 8)

| # | Input | Range |
|---|---|---|
| 0–4 | ray distance / R (rays at −60°, −30°, 0°, +30°, +60°) | [0, 1] |
| 5–6 | sin, cos of goal bearing relative to heading | [−1, 1] |
| 7 | goal distance / 10, clipped | [0, 1] |
| 8–9 | previous left / right wheel command ("memory") | [−1, 1] |

Configurable in `SimParams` (saved with the weights): `ray_angles_deg`, `goal_inputs` (`sincos_dist` | `sincos` | `bearing`),
`prev_action_inputs`, `reverse_max`. No observation normalization wrapper, so exported weights need no extra statistics.

### Action (2 outputs)

Two outputs → `uL, uR`, **linear layer clipped to [−1, 1]** (the standard SB3 Box-action setup; deterministic action = clip(mean)).
Per-wheel sign is free (spin in place), but **body speed is floored at 0 — no reversing** (see decisions log).

### Network

Leading candidate: **10 → 12 → 12 → 4 → 2**, tanh hidden layers (~350 parameters) — every neuron still drawable, and the
4-neuron layer before the wheels reads well in the network panel. Final choice between the two best variants
(plain, or trained with a privileged critic — same shipped actor) pending a multi-seed comparison (§0, §11).
Original design was 8 → 6 → 6 → 2 (110 parameters); a GRU actor (`recurrent.py`) was tried and did not help.

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
    geometry.py  sim.py  env.py  distance_field.py  mapgen.py  render.py   # sim + env (ported to TS in M3)
    train.py  vec.py  evaluate.py  baselines.py                             # PPO, batched vec env, test set, baselines
    policy.py (MLP export/load)  recurrent.py (GRU)  privileged.py (critic-only inputs)
  presets/                  # demo preset layouts (JSON), shared by Python and TS
  scripts/                  # make_presets, render_samples, bench_env, progress, rollouts, analyze_mazes
  tests/                    # geometry, sim, distance field, mapgen, env/reward, export parity, privileged, GRU
  configs/                  # one yaml per experiment (see PLAN §11 for results)
  artifacts/                # runs/<run>/ (config, weights, validation, eval reports), eval/ (baselines), samples/
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
  showNetwork?: boolean;          // neuron panel (default true)
  version?: string;               // starting Wheely: heuristic | v3-rookie | v4-owl-eyes | final (default)
  sprite?: string;                // URL of the mascot sprite (site asset, not in this MIT package)
}): { destroy(): void };
```

- Theme: reads `--fg`, `--bg`, `--surface`, `--border`, `--muted`, `--accent`, `--accent-fg`, `--ai` (negative signals), `--radius`,
  `--font-mono` from the container (re-read on theme change). No theme props.
- The demo owns its layout inside `el` (arena, controls, network panel; panel moves below the arena on phones).
  The site's fixed-aspect `.stage` box will need loosening when the real demo is wired in.
- Pauses when off-screen or tab hidden; respects reduced motion.
- `mountMascot` is **dropped** (no roaming mascot in the site hero).
- Site wiring (self-contained, Bryan 2026-10-04): the site keeps a generated copy of the demo build in
  `src/vendor/maze-bot/`; `npm run sync:site` here rebuilds and copies it (commit stamped in `SOURCE.txt`).
  `src/demos/maze.ts` re-exports it. No npm dependency or tag needed.

## 8. Milestones

Each milestone ends with a check run by Claude (tests, metrics, screenshots) and a short report to Bryan before moving on.

| # | Deliverable | Done when |
|---|---|---|
| M0 ✅ | `uv` project (uv installs Python 3.12), git repo, CI skeleton, CLAUDE.md, README stub | `uv run pytest` + `uv run ruff check` green locally; CI file ready (runs once Bryan pushes) |
| M1 ✅ | Geometry (capsules, rays, collision), kinematics, env, distance field, map generator (all categories + presets), matplotlib renderer | Unit tests + `check_env` pass; rendered sample sheets of every category reviewed; env steps/s measured |
| M2 ✅ | PPO training + curriculum + held-out evaluation + baselines | Per-category held-out success rates, failure taxonomy, rollout GIFs; hidden size decided with Bryan |
| M3 ✅ | `weights.json` export + TS sim port + parity tests | Parity tests pass locally and in CI |
| M4 ✅ | `mountMazeDemo`: canvas, presets, draw mode, path-check warning, network panel, controls | Works in the dev page; screenshots desktop + 390 px, light + dark, reviewed |
| M5 (in review) | Site wiring: self-contained copy via `npm run sync:site`; `/playground` is the demo | Site `npm run check:all` passes; Bryan reviews `npm run dev` + `npm run preview` |
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
- [x] License: MIT for code; mascot art stays © Bryan Chew.
- [ ] "Train it live" (in-browser neuroevolution) — later, after M6.

## 11. Decisions log

- **2026-10-04 — Plan updated** for: Wheely naming, no roaming mascot (`mountMascot` dropped), freehand walls without snapping
  plus presets, "no path" warning that still lets Wheely run, 8 → 6 → 6 → 2 with every neuron visualised.
- **2026-10-04 — Train on random shapes, not strict mazes** (drawing is the main interaction; failures should come from
  missing memory, not OOD shapes). Mazes-only vs mixed ablation in M2. Varying arena size, clipped linear outputs,
  per-step parity accepted.
- **2026-10-05 — M2 debugging (each finding from a 3M-step probe, validation = deterministic actor):**
  1. *Blind reversing.* With reverse allowed, the policy learned the near-linear controller
     speed ∝ cos(goal bearing): goal behind ⇒ drive backwards, blind (rays only look forward).
     Training noise jostled it free, so train reward rose while validation fell 0.67 → 0.47.
     A 30% reverse cap didn't help (still reversed 97% of steps). **Fix: no backward body motion**
     (a circular robot can always spin free, so reversing is never needed).
  2. *Freezing.* A wall-contact penalty of 0.2 made "stop when unsure" optimal once mazes entered the
     curriculum (grinding = −0.21/step vs standing still = −0.01/step); validation collapsed 0.76 → 0.50.
     **Fix: contact penalty back to 0.05.** Result: 0.76 → 0.80 → 0.82 at 1/2/3M steps.
  3. **Exploration-noise cap** annealed 0.6 → 0.1 over 6M steps, so the shipped deterministic policy
     can't rely on noise to escape dead-bands. (Didn't fix 2 on its own; kept as a safeguard.)
- **2026-10-05 — M2 target: match or beat the hand-coded reactive baseline** on the test set
  (1,000 maps/category; "match" = within 2 pts in every category). First 6-6 run: level on open/obstacles/
  scribbles/mixed, behind on mazes (44% vs 69%), presets (81% vs 94%), traps (18% vs 38%). Bigger nets
  (8-8, 64-64) did not help ⇒ next: wider vision, 10 inputs (7 rays + goal sin/cos/dist), 12-12 hidden —
  side view (0, ±30, ±60, ±90°) and 360° view (every ~51°), plus a 5-ray 12-12 control. If that falls
  short: one more hidden layer (12-12-12).
- **2026-10-05 — M2 experiment round (test set, 1,000 maps/category vs reactive baseline; one seed each):**
  - Network size/depth with 5 forward rays (6-6, 8-8, 12-12, 12-12-12, 64-64): mazes 43-46%, no gain.
  - Vision: 360° (7 rays) lifts mazes to 56% and traps to 64%; side-only rays hurt. 360° + memory: no extra gain.
  - **Memory (previous wheel commands as inputs) is the biggest win**: 12-12 and 12-12-4 match/beat the baseline in
    6 of 7 categories (traps 56-76% vs 38%; scribbles, obstacles, mixed above baseline); mazes 57-60% vs 69%.
  - No gain (or worse): 12-12-12, 12-12-8-4, gentle maze curriculum, reversing (+/- penalty; blind reversing
    returns without one), dropping goal distance or using a single bearing input, privileged critic (distance:
    level; + path direction: worse), GRU(8) recurrent actor (mazes 54.7%; trained on GPU with a fused-GRU path).
  - Maze failure analysis: the learned policy matches the baseline on short/direct mazes but collapses with
    path turns (6-8 turns: 3-5% vs 44%); 84% of failures occur in the first quarter of the path ("ping-pong"
    in corridors leading away from B). The policy goal-seeks well but will not commit to long detours.
  - Also no gain: speed bonus on reaching B (+10 always, up to +5 for finishing early; mazes 57.0%, traps 48.6%,
    ~0.2 s faster) and longer horizon γ = 0.995 (mazes 53.3%, winding mazes unchanged). Trap success varies
    32-76% across similar runs — treat single-seed trap numbers with caution.
  - Validation with 100 maps/category is noisy; best-checkpoint selection inflates validation scores (GRU:
    val mazes 0.60 vs test 0.547). Use larger validation sets for final comparisons.
- **2026-10-05 — M2 closed. Final policy = memory 12-12-4 trained with a privileged critic** (best on mazes,
  presets, mixed; same shipped actor as the plain 12-12-4). Multi-seed comparison skipped (Bryan's call); the
  write-up should say the pick is from single-seed runs. Release bundle in `artifacts/release/`.
- **2026-10-04 — MIT license** (code); mascot art excluded. Python env = uv venv in `.venv/`.
- **2026-10-04 — Demo versions and UX (with Bryan):** playable: By-the-Book (heuristic), Rookie (`mix_6x6`), Owl Eyes (`mix_360_12x12`),
  Wheely (final). Write-up only: Moonwalker (v0, clip on Open field), Scaredy-Wheely (v2, clip on Slalom; freezes in 24% of empty
  arenas with no wall contact), one-step memory and GRU (numbers only). Bundle + test-set reports: `artifacts/release/versions/`
  (`scripts/make_versions.py`). Write-up clips are recorded from the web demo, not matplotlib. Site layout and brain panel: site
  PLAN §14 (2026-10-04). The network panel adapts to each version's arch (8-6-6-2, 10-12-12-2, 10-12-12-4-2; rule view for
  By-the-Book), so §1's "8 → 6 → 6 → 2" is superseded.
