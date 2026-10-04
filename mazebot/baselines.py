"""Hand-written baselines that see exactly what the learned policy sees (the 8-dim observation)."""

from __future__ import annotations

import math

import numpy as np

from .sim import SimParams


class RandomPolicy:
    """Uniform random wheel commands, held for 0.5 s so it actually goes somewhere."""

    def __init__(self, seed: int = 0, hold: int = 8):
        self.rng = np.random.default_rng(seed)
        self.hold = hold
        self.t = 0
        self.a = np.zeros(2)

    def __call__(self, obs):
        if self.t % self.hold == 0:
            self.a = self.rng.uniform(-1, 1, 2)
        self.t += 1
        return self.a


class ReactivePolicy:
    """Goal-seeking with reactive obstacle avoidance (memoryless, like the learned policy).

    Steer towards the goal bearing; slow down and turn towards the more open side when the front
    rays are short; while a wall is close on the goal side, keep it there instead of turning into it
    (a crude, memoryless wall-follow).
    """

    def __init__(self, sim: SimParams | None = None):
        self.sim = sim or SimParams()

    def __call__(self, obs):
        p = self.sim
        rays = np.asarray(obs[:5]) * p.ray_range
        bearing = math.atan2(obs[5], obs[6])  # > 0: goal is to the right (clockwise)
        left = min(rays[0], rays[1])
        right = min(rays[3], rays[4])
        front = min(rays[1], rays[2], rays[3])

        turn = float(np.clip(1.5 * bearing, -1.0, 1.0))
        # don't steer into a close wall on the goal side
        if turn > 0 and right < 0.7:
            turn = min(turn, (right - 0.5) * 2.0)
        if turn < 0 and left < 0.7:
            turn = max(turn, -(left - 0.5) * 2.0)
        if front < 1.0:
            side = 1.0 if right > left else -1.0
            turn = side * min(1.0, 0.4 + (1.0 - front))
        speed = float(np.clip((front - 0.4) / 0.8, 0.0, 1.0)) * (1.0 - 0.5 * abs(turn))
        ul = np.clip(speed + 0.6 * turn, -1.0, 1.0)
        ur = np.clip(speed - 0.6 * turn, -1.0, 1.0)
        return np.array([ul, ur])
