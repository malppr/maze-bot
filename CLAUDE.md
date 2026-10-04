# maze-bot

"Wheely's maze": RL navigation for the portfolio's mascot robot. Python (uv) for the env and training,
TypeScript (`web/`, from M3) for the browser port and demo. Read `PLAN.md` first (design, milestones,
decisions log — §11 has every experiment and its result). The shipped policy and the sim spec for the
TS port are in `artifacts/release/` (README.md there). Next work: web wiring, see `HANDOFF.md`.

## Environments

- `.venv/` (main, CPU): `uv sync` installs Python 3.12 + dev + train groups (CPU torch, sb3). Use `uv run …`.
  There is no system Python; never install into a global Python.
- `.venv-gpu/` (GPU, not managed by pyproject): CUDA 12.8 torch 2.11 + sb3-contrib, for RecurrentPPO/GRU runs.
  Run with `.venv-gpu/Scripts/python.exe -m …`. Recreate: `uv venv .venv-gpu --python 3.12`, then
  `uv pip install --python .venv-gpu/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128`,
  then numpy gymnasium stable-baselines3 sb3-contrib tensorboard pyyaml matplotlib imageio pytest, then `-e . --no-deps`.
- Never `uv sync` / change torch in `.venv` while a training is running (Windows locks the DLLs; queued
  `uv run` jobs would re-sync). CI-like check without touching `.venv`:
  `UV_PROJECT_ENVIRONMENT=.venv-ci uv sync --no-default-groups --group dev` then `uv run --no-sync pytest`.

## Commands

- `uv run pytest` / `uv run ruff check` / `uv run ruff format` (CI: ruff check + format --check + pytest, dev group only)
- Train: `uv run python -m mazebot.train configs/<cfg>.yaml [--steps N] [--name NAME] [--hidden 12,12] [--resume model.zip --start-stage K --patience N]`
  → `artifacts/runs/<name>/` (config, weights.json = best validation checkpoint, validation.jsonl, tb/, ckpt/).
  GRU configs (`recurrent:` + `device: cuda`) need `.venv-gpu`.
- Test set: `uv run python -m mazebot.evaluate --policy artifacts/runs/<run>/weights.json --n 1000 --workers 12`
  (also `--policy reactive|random`) → `<run>/eval/report.{md,json}`, episodes.jsonl.
- Analysis: `scripts/analyze_mazes.py <run>/eval` (maze difficulty vs success, vs baseline),
  `scripts/progress.py <run>` (validation curves + trajectories per checkpoint), `scripts/rollouts.py <weights>` (sheets + GIFs).
- Other: `scripts/render_samples.py` (map sheets), `scripts/make_presets.py` (rebuild presets.json), `scripts/bench_env.py`.
- Web (`web/`, Node 24 at `C:\Program Files\nodejs`): `npm ci`, `npm test` (vitest parity tests), `npm run typecheck`.
- Site: `npm run dev` (demo dev page, port 5180); `npm run sync:site` copies the demo build into `../malppr.github.io/src/vendor/maze-bot/`
  (then commit it in the site repo). Commit here first so `SOURCE.txt` names a clean commit.
  Regenerate parity fixtures after any sim/policy change: `uv run python scripts/make_parity_fixtures.py` → `web/test/fixtures/`.
- Versions bundle (demo Wheelys + test-set reports): `uv run python scripts/make_versions.py --skip-gifs` → `artifacts/release/versions/`.
- TensorBoard: `uv run tensorboard --logdir artifacts/runs` → http://localhost:6006

## Rules

- Bryan runs `git push` himself. Commit locally and say what is ready to push; never push.
- **Compute: at most two trainings at a time** (two saturate the 8-core CPU). Don't add load (parallel runs,
  more envs/threads) without asking. Queue extra runs to start when a slot frees.
- Each milestone ends with a check (tests, metrics, screenshots) and a short report before moving on.
- Discuss scope/design changes first. Report results honestly: test set (1,000 maps/category, `EVAL_SEED`),
  never the validation set (`VAL_SEED`, 100 maps/category, used for checkpoint selection and early stopping).
- Python and TS sims must stay in lock-step: any change to `mazebot/geometry.py`, `sim.py`, `env.py` (observation)
  or `distance_field.py` needs the matching change in `web/src/sim/` and passing parity tests.
- Conventions: x right, y down, heading θ clockwise-positive on screen, float64 everywhere.
- Windows: `Monitor`/`tail -F` watchers leave orphaned `tail`/`grep` processes when stopped; clean them up.
