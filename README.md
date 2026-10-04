# Wheely's maze (maze-bot)

A small two-wheeled robot learns with reinforcement learning (PPO) to drive from A to B. Its brain is a
350-parameter network (10 → 12 → 12 → 4 → 2): five forward distance rays, the direction and distance to B, and
its previous wheel command in; two wheel speeds out, 15 times a second.

**[Play with it](https://malppr.github.io/playground)** (draw walls, switch versions, watch every neuron) ·
**[Write-up](https://malppr.github.io/projects/wheelys-maze)**

## Results

Success rate (%) on 1,000 held-out layouts per category. By-the-Book is a hand-written controller that sees the
same inputs.

| | open | obstacles | scribbles | mixed | mazes | traps |
|---|---|---|---|---|---|---|
| By-the-Book (hand-written) | 100.0 | 95.6 | 89.7 | 84.6 | **68.6** | 38.4 |
| Rookie (8 inputs, 8-6-6-2) | 100.0 | 93.9 | 87.2 | 82.4 | 43.6 | 18.2 |
| Owl Eyes (7 rays all round) | 99.3 | 95.3 | 91.2 | 83.1 | 56.3 | **63.8** |
| **Wheely** (memory inputs, shipped) | **100.0** | **96.8** | **92.9** | **87.6** | 59.7 | 60.7 |

Matches or beats the hand-written rule in 6 of 7 categories (presets: 94.0 vs 94.1); the gap is winding mazes,
where it won't commit to long detours away from B. Single seed per configuration. Every run and experiment:
[PLAN.md §11](PLAN.md), [artifacts/release/](artifacts/release/README.md) (weights, sim spec, test reports,
maze analysis), [all runs](artifacts/release/all_runs_test_set.md).

## Layout

- `mazebot/`: Python simulator (capsule geometry, kinematics, rays), Gymnasium env, map generator, training
  (stable-baselines3 PPO, curriculum), evaluation.
- `web/`: TypeScript port of the simulator and policies, and the demo (`mountMazeDemo`). Parity-tested
  against Python (worst error over 486-step closed-loop rollouts: 3e-14).
- `artifacts/release/`: the shipped policy and every demo version with its test-set report.

## Commands

```sh
uv sync                    # Python 3.12 env in .venv (uv-managed)
uv run pytest              # Python tests;  uv run ruff check
uv run python -m mazebot.train configs/<cfg>.yaml
uv run python -m mazebot.evaluate --policy <weights.json> --n 1000

npm ci && npm test         # TypeScript parity tests (fixtures: scripts/make_parity_fixtures.py)
npm run dev                # standalone demo page on http://localhost:5180
npm run sync:site          # copy the demo build into the portfolio site (../malppr.github.io)
```

## License

Code: MIT (see [LICENSE](LICENSE)). The Wheely mascot artwork is © Bryan Chew and not MIT-licensed.
