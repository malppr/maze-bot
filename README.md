# Wheely's maze (maze-bot)

Wheely, a small two-wheeled robot, learns with reinforcement learning to drive from A to B using five
distance rays and the direction to the goal. Its policy is a tiny 8 → 6 → 6 → 2 network that runs in the
browser, where you can freehand-draw walls and watch every neuron.

Work in progress — see [PLAN.md](PLAN.md).

## Development

The project uses a uv-managed virtual environment in `.venv/` (Python 3.12, installed by uv).

```sh
uv venv --python 3.12   # create .venv (uv sync also creates it if missing)
uv sync                 # install dependencies into .venv
uv run pytest           # tests
uv run ruff check       # lint
uv run python scripts/render_samples.py   # sample maps per category -> artifacts/samples/
```

## License

Code: MIT (see [LICENSE](LICENSE)). The Wheely mascot artwork is © Bryan Chew and not MIT-licensed.
