# Handoff — Wheely's maze (maze-bot) and the portfolio

Prompt for the next Claude instance. Paste everything below the line.

---

You are continuing work with Bryan on his portfolio. Two repos:

- `D:\Proj\malppr.github.io` — the portfolio site (Astro, live on GitHub Pages via Actions on push to `main`).
  Read its `CLAUDE.md` and `PLAN.md` (§8 integration contract, §14 decisions log).
- `D:\Proj\maze-bot` — "Wheely's maze": a small two-wheeled robot (the site mascot, Wheely) trained with PPO to drive
  from A to B; ships to the site's Playground as a browser demo. **Read `CLAUDE.md` and `PLAN.md` first** —
  §0 is the status, §11 the decisions log with every experiment and its result.

## How Bryan works (non-negotiable)

- He runs `git push` himself. Commit locally (end commit messages with the Co-Authored-By line) and tell him what is
  ready to push; never push. maze-bot `main` is ~30 commits ahead of `origin/main`.
- Nothing goes live without his local review (`npm run dev`, then `npm run preview`; `npm run check:all` must pass)
  before he merges to the site's `main`. Site work goes on a branch.
- Discuss big design/scope decisions first; show options or mockups for anything visual. Verify visual changes with
  screenshots at desktop and 390 px, light and dark (Playwright with his Edge works).
- **Compute: at most two training runs at a time** (two saturate his 8-core Ryzen 7 7700). Don't add load (more
  parallel jobs, envs, threads) without asking; queue runs to start when a slot frees.
- Report honestly: test-set numbers (1,000 maps per category), failure modes, what didn't work.
- Public content is personal/academic only (no SAP or Bosch work, no CV on the site).
- Each milestone ends with a check you run yourself and a short report before moving on.

## Environment

Windows 11, Git Bash + PowerShell. Node 24 at `C:\Program Files\nodejs` (add to PATH in Bash). uv manages Python
(no system Python). maze-bot has two envs: `.venv` (CPU, `uv run …`) and `.venv-gpu` (CUDA 12.8 torch + sb3-contrib,
for GRU/RecurrentPPO; RTX 5060 Ti 16 GB, Blackwell sm_120 works). Never `uv sync`/change torch in `.venv` while a
training runs (Windows DLL locks). `gh` is not installed. TensorBoard: `uv run tensorboard --logdir artifacts/runs`.

## Where things stand (2026-10-05)

- **M0, M1 done** (env, capsule geometry, maps in 7 categories + 6 demo presets, 66 tests, CI file).
- **M2 nearly done.** ~25 PPO runs. Key facts:
  - Fixed along the way: blind reversing (now no backward motion), freezing from a too-high wall-contact penalty
    (0.05 now), exploration-noise cap annealed 0.6 → 0.1.
  - **Best policies** (test set vs hand-coded reactive baseline): memory inputs (previous wheel commands) +
    actor 10 → 12 → 12 → 4 → 2 — `artifacts/runs/mix_fwd5mem_12x12x4_s0` (traps 75.7%) and
    `artifacts/runs/mix_fwd5mem_12x12_pcrit_s0` (same actor, trained with a privileged critic; mazes 59.7%).
    Both match/beat the baseline in 6 of 7 categories; **mazes plateau at ~55-60% vs 69%.**
  - Didn't help: bigger/deeper nets, 360° or side rays (360° alone helped mazes/traps before memory), mazes-only
    or gentle curriculum, reversing ± penalty, fewer goal inputs, critic with path direction, GRU(8) actor,
    speed bonus, γ = 0.995. See PLAN §11.
  - Maze analysis (`scripts/analyze_mazes.py`, `artifacts/runs/mix_fwd5mem_12x12_pcrit_s0/eval/analysis_mazes.md`):
    learned policy equals the baseline on direct mazes, collapses with path turns (6-8 turns: ~5% vs 44%);
    failures are corridor "ping-pong" early in the path — it won't commit to long detours away from B.
  - Validation (100 maps/category) is noisy and best-checkpoint selection inflates it; trap results vary 32-76%
    across similar runs. Use more seeds and more validation maps for final comparisons.

## Next steps, in order

1. **Close M2.** Train the two candidate configs (`configs/mix_fwd5mem_12x12x4.yaml`,
   `configs/mix_fwd5mem_12x12_pcrit.yaml`) with `--seed 1` and `--seed 2` (two at a time), with validation raised
   to 300 maps/category for these runs; evaluate all on the test set; pick the final policy by mean ± spread.
   Make rollout GIFs (`scripts/rollouts.py`) and progress plots; report to Bryan for sign-off. Copy the chosen
   `weights.json` to a stable path (e.g. `artifacts/release/weights.json`).
2. **Part 4 planning (plan only, discuss with Bryan):** (a) PushT diffusion-policy reproduction (state-based) +
   MLP-BC baseline + ablations, ONNX export as an advanced Playground level (GPU env pattern above works);
   stretch: Wheely pushes the T. (b) Robot-arm project page (currently hidden) with an episode replay viewer
   (URDF + three.js). (c) One AI-only side project — propose 2-3 options.
3. **M3:** `weights.json` export is done; port the sim to TypeScript (`web/src/sim/`) and add Python→TS parity
   fixtures and vitest tests (per-step ≤ 1e-9, 300-step rollouts ≤ 1e-6, policy forward ≤ 1e-6). Port exactly:
   `geometry.py`, `sim.py` (incl. `reverse_max`, goal-input modes), env observation incl. previous-action memory.
   Root `package.json` exporting `maze-bot/web`; CI gets a node job.
4. **M4** demo (`mountMazeDemo`: presets, freehand draw, no-path warning, network panel — mock up options first),
   **M5** v0.1.0 + site wiring (`src/demos/maze.ts`, `IS_PREVIEW = false`, loosen `.stage` aspect box; site PLAN §8
   still lists the dropped `mountMascot` — fix there), **M6** write-up + site project page (tags robot-learning + ai;
   the maze-gap analysis is the honest centrepiece).

## Gotchas

- `cmd | tail -1 && git commit` commits even when the command failed — check exit codes.
- Stopped Monitor/`tail -F` watchers leave orphan `tail`/`grep` processes on Windows; clean them up.
- A killed training leaves no `weights.json` (only `val/step_*.json` checkpoints).
- CRLF warnings on commit are harmless (`.gitattributes` normalises to LF).
