# maze-bot

"Wheely's maze": RL navigation for the portfolio's mascot robot. Python (uv) for the env and training,
TypeScript (`web/`) for the browser port and demo. Read `PLAN.md` first (design, milestones, decisions log).

## Commands

- `uv venv --python 3.12` then `uv sync` — project venv in `.venv/` (uv manages Python; there is no system Python).
  Always run things through `uv run …` (or `.venv/Scripts/python`); never install into a global Python.
- `uv run pytest` — tests; `uv run ruff check` / `uv run ruff format` — lint/format
- `uv run python scripts/render_samples.py` — render sample maps of every category

## Rules

- Bryan runs `git push` himself. Commit locally and say what is ready to push; never push.
- Each milestone ends with a check (tests, metrics, screenshots) and a short report before moving on.
- Discuss scope/design changes first. Report results honestly (held-out, per category, with failure modes).
- Python and TS sims must stay in lock-step: any change to `mazebot/geometry.py`, `env.py` or `distance_field.py`
  needs the matching change in `web/src/sim/` and passing parity tests.
- Conventions: x right, y down, heading θ clockwise-positive on screen, float64 everywhere.
