"""Build presets/presets.json: the demo's preset layouts (landscape 10 x 6.25; portrait = transposed).

Shared by the Python evaluation ("presets" category) and the browser demo. Re-run after editing.
"""

import json
from pathlib import Path

import numpy as np

from mazebot import mapgen

W, H = 10.0, 6.25
R = 0.1  # wall radius


def bar(ax, ay, bx, by, rad=R):
    return [ax, ay, bx, by, rad]


def post(x, y, rad):
    return [x, y, x, y, rad]


def small_maze():
    rng = np.random.default_rng(8)
    nx, ny = 6, 4
    segs = mapgen.maze_walls(rng, nx, ny, 1.0, braid=0.08) * np.array([W / nx, H / ny, W / nx, H / ny])
    return mapgen.wobble(rng, segs, 0.08, 0.08, amp=0.03).round(4).tolist()


PRESETS = [
    {"name": "Open field", "walls": [], "start": [1.2, 3.125, 0.0], "goal": [8.8, 3.125]},
    {
        "name": "Posts",
        "walls": [
            post(x, y, r)
            for x, y, r in [
                (2.6, 1.6, 0.35),
                (3.4, 4.3, 0.45),
                (4.8, 2.6, 0.3),
                (5.6, 5.0, 0.3),
                (6.4, 1.3, 0.4),
                (7.2, 3.6, 0.45),
                (8.4, 1.9, 0.25),
                (1.6, 4.9, 0.3),
            ]
        ],
        "start": [0.9, 1.0, 0.4],
        "goal": [9.1, 5.3],
    },
    {
        "name": "Slalom",
        "walls": [bar(2.5, 0, 2.5, 4.2), bar(5.0, 2.05, 5.0, 6.25), bar(7.5, 0, 7.5, 4.2)],
        "start": [1.1, 1.2, 1.5708],
        "goal": [8.9, 1.2],
    },
    {
        "name": "Rooms",
        "walls": [
            bar(3.4, 0, 3.4, 2.2),
            bar(3.4, 3.6, 3.4, 6.25),
            bar(6.6, 0, 6.6, 3.9),
            bar(6.6, 5.2, 6.6, 6.25),
            bar(3.4, 2.2, 4.6, 2.2),
            bar(5.0, 3.4, 5.0, 6.25),
        ],
        "start": [1.4, 1.2, 0.0],
        "goal": [8.6, 1.4],
    },
    {"name": "Small maze", "walls": small_maze(), "start": [0.8, 0.8, 0.0], "goal": [9.2, 5.45]},
    {
        "name": "The trap",
        "walls": [bar(4.4, 1.6, 6.6, 1.6), bar(6.6, 1.6, 6.6, 4.65), bar(6.6, 4.65, 4.4, 4.65)],
        "start": [1.4, 3.125, 0.0],
        "goal": [8.4, 3.125],
    },
]

out = {
    "version": 1,
    "note": "Landscape, world units, x right, y down. Portrait = swap x, y. Walls: [ax, ay, bx, by, rad].",
    "presets": [{"width": W, "height": H, **p} for p in PRESETS],
}
path = Path(__file__).resolve().parent.parent / "presets" / "presets.json"
path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {len(PRESETS)} presets to {path}")
