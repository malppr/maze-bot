# Handoff — put Wheely's maze on the website (M3–M6)

Prompt for the next Claude instance. Paste everything below the line.

---

You are continuing work with Bryan on his portfolio. Your job: **take the trained "Wheely's maze" policy and put it
live on his site as an interactive demo plus a project page** — maze-bot milestones M3 → M6.

## Repos — read these first

- `D:\Proj\maze-bot` — the RL project. Read `CLAUDE.md`, `PLAN.md` (§0 status, §1 visitor experience, §3 freehand
  walls, §7 integration contract, §8 milestones, §11 decisions log) and **`artifacts/release/README.md`** (final
  weights, results, and the exact sim/observation spec your TypeScript must reproduce).
- `D:\Proj\malppr.github.io` — the site (Astro, deployed by GitHub Actions on push to `main`). Read `CLAUDE.md`,
  `PLAN.md` (§8 demo contract, §14 decisions). Demo swap point: `src/demos/maze.ts` (re-exports the scripted stub
  `maze-stub.ts`, `IS_PREVIEW = true`); page: `src/pages/playground.astro`; mascot sprite
  `src/assets/mascot/mascot-top.svg`; project pages in `src/content/projects/<slug>/index.mdx`.

## How Bryan works (non-negotiable)

- He runs `git push` himself (including tags). Commit locally, end commit messages with the Co-Authored-By line,
  and tell him what is ready to push; never push.
- Site work happens on a branch. Nothing goes live without his local review: `npm run dev`, then
  `npm run preview`; `npm run check:all` must pass before he merges to `main`.
- Discuss design/scope decisions first; **show 2-3 options or mockups for anything visual** before building.
- Verify visual changes with screenshots at desktop and 390 px width, light and dark (Playwright with his Edge).
- Don't raise compute load without asking (at most two heavy jobs at a time).
- Report honestly. Public content is personal/academic only (no SAP/Bosch, no CV).
- Each milestone ends with a check you run yourself and a short report to him before moving on.

## Environment

Windows 11, Git Bash + PowerShell. Node 24 at `C:\Program Files\nodejs` (add to PATH in Bash). uv manages Python
(no system Python): maze-bot `.venv` via `uv run …`; `.venv-gpu` only for GRU training (not needed now).
`gh` is not installed. Bryan pushes; check `git status -sb` in both repos for what is unpushed (site repo was clean).

## The policy you are shipping

`maze-bot/artifacts/release/weights.json`: actor 10 → 12 → 12 → 4 → 2 (tanh hidden, linear output clipped),
350 parameters. Inputs: 5 forward rays / 3, sin & cos of goal bearing, goal distance / 10, previous left/right
wheel command (memory). Matches/beats a hand-coded controller in 6 of 7 test categories; mazes are the honest gap
(59.7% vs 68.6%). Reaches B on 5 of 6 presets; loops inside "The trap" (intended demo moment). No reversing
(body speed floored at 0), spin in place allowed.

## Plan

**M3 — TypeScript port + parity (in maze-bot `web/`).**
- `web/src/sim/`: geometry (capsules, rays, push-out), sim (physics step, observation incl. memory), reachability
  grid for the "no path" warning, policy inference from `weights.json`. Follow `artifacts/release/README.md` and
  the Python sources line by line (only + − × ÷ sqrt in geometry; x right / y down; float64).
- Add a Python script that writes parity fixtures (random states → one physics step; ray casts; observations;
  policy forward; 300-step closed-loop rollouts on a few seeded maps and presets) and vitest tests against them:
  per-step ≤ 1e-9, rollouts ≤ 1e-6, policy forward ≤ 1e-6.
- Root `package.json` (npm git deps need it at the root) exporting `maze-bot/web`; add a node job to CI.
- Check: parity tests pass locally; report.

**M4 — `mountMazeDemo(el, { mode, showNetwork }) → { destroy() }`** (contract: maze-bot PLAN §7).
- First mock up 2-3 layouts for Bryan (arena + controls + network panel; desktop and 390 px; light and dark).
  The panel must show every neuron (10, 12, 12, 4, 2) with live values, edges by sign/|w|, hover to highlight.
- Features (PLAN §1): presets (`presets/presets.json`; portrait = transpose on phones), freehand drawing (pointer →
  RDP → capsules, eraser, clear; brush radius ~0.08-0.12), drag A/B, "No path to B — Wheely will get lost" warning
  that still runs, "lost" state after the timeout, speed 0.5×/1×/4×, pause/step/reset, rays drawn, sprite rotated
  θ + π/2. Theme from CSS vars `--fg --bg --surface --border --muted --accent` (re-read on theme change). Pause
  off-screen / hidden tab; reduced motion → start paused. Standalone Vite dev page in `web/dev/`.
- Check: works in the dev page on desktop and phone; screenshots reviewed by Bryan.

**M5 — release + site wiring.**
- Propose to Bryan how the site consumes the package (e.g. `"maze-bot": "github:malppr/maze-bot#v0.1.0"` with a
  built ESM bundle + types, vs. TypeScript source compiled by the site's Vite) and agree before tagging.
- Site branch: `src/demos/maze.ts` → `export { mountMazeDemo } from 'maze-bot/web'`, `IS_PREVIEW = false`; update the
  Playground copy (it still says "You'll be able to draw…"); loosen the fixed-aspect `.stage` box so the demo owns
  its layout; update site PLAN §8 (drop `mountMascot`, which was dropped 2026-10-04) and add §14 entries.
- Check: `npm run check:all` passes, Lighthouse on /playground stays ≥ 95, screenshots; Bryan reviews dev + preview;
  tag `v0.1.0` ready for him to push.

**M6 — write-up.** maze-bot README + site project page "Wheely's maze" (tags `robot-learning`, `ai`; `demo: maze`;
ask whether it is featured). Honest story from PLAN §11 and `artifacts/release/`: method, reward design, the
bugs PPO exploited (blind reversing, freezing), what helped (memory) and what didn't, the maze-gap analysis,
single-seed caveat. Media from `artifacts/release/gifs/` (convert to short webm/mp4 per the site's video budget).

**After M6:** Part 4 planning (PushT diffusion policy as an advanced Playground level, the hidden robot-arm page with
a replay viewer, one AI-only side project — 2-3 options). Plan only, with Bryan.

## Gotchas

- `cmd | tail -1 && git commit` commits even when the command failed — check exit codes.
- Stopped Monitor/`tail -F` watchers leave orphan `tail`/`grep` processes on Windows; clean them up.
- CRLF warnings on commit are harmless (`.gitattributes` normalises to LF).
- Don't `uv sync` or change torch in maze-bot `.venv` while anything uses it (Windows DLL locks).
