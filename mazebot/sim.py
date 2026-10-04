"""Robot kinematics, collision handling and observations (ported 1:1 to `web/src/sim/sim.ts`).

Conventions: x right, y down (screen). Heading th = 0 points along +x; positive th turns clockwise on
screen. Wheel commands uL, uR are in [-1, 1].
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import geometry


@dataclass(frozen=True)
class SimParams:
    radius: float = 0.3  # robot collider radius
    wheel_base: float = 0.6  # distance between wheels
    v_max: float = 1.2  # wheel surface speed at u = 1 (units/s)
    reverse_max: float = 0.0  # no backward motion: rays only look forward; a circle can always spin free
    dt: float = 1.0 / 30.0  # physics step
    frame_skip: int = 2  # physics steps per policy step (policy at 15 Hz)
    ray_angles_deg: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)
    ray_range: float = 3.0
    goal_dist_scale: float = 10.0  # obs[7] = min(goal distance / scale, 1)
    goal_radius: float = 0.35  # success when the centre is this close to B
    prev_action_inputs: bool = False  # append the previous (uL, uR) to the observation: one step of memory

    ray_angles: tuple[float, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "ray_angles", tuple(math.radians(a) for a in self.ray_angles_deg))

    @property
    def obs_dim(self) -> int:
        return len(self.ray_angles) + 3 + (2 if self.prev_action_inputs else 0)

    @classmethod
    def from_dict(cls, d: dict | None) -> SimParams:
        d = dict(d or {})
        if "ray_angles_deg" in d:
            d["ray_angles_deg"] = tuple(float(a) for a in d["ray_angles_deg"])
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("ray_angles")
        d["ray_angles_deg"] = list(self.ray_angles_deg)
        return d


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def physics_step(x: float, y: float, th: float, ul: float, ur: float, caps: np.ndarray, p: SimParams):
    """Advance one physics step. Returns (x, y, th, contact).

    Differential drive: v = v_max (uL + uR)/2, w = v_max (uL - uR)/b (y-down, so left faster => clockwise).
    Body speed is floored at -reverse_max * v_max (0: no reversing; spinning in place is unaffected).
    Midpoint integration; sub-stepped so the centre never moves more than r/2 per sub-step.
    """
    v = max(p.v_max * 0.5 * (ul + ur), -p.reverse_max * p.v_max)
    w = p.v_max * (ul - ur) / p.wheel_base
    n = max(1, math.ceil(abs(v) * p.dt / (0.5 * p.radius)))
    h = p.dt / n
    contact = False
    for _ in range(n):
        tm = th + 0.5 * w * h
        x = x + v * math.cos(tm) * h
        y = y + v * math.sin(tm) * h
        th = wrap_angle(th + w * h)
        x, y, c = geometry.push_out(x, y, p.radius, caps)
        contact = contact or c
    return x, y, th, contact


def ray_dirs(th: float, p: SimParams) -> np.ndarray:
    return np.array([[math.cos(th + a), math.sin(th + a)] for a in p.ray_angles])


def observe(x: float, y: float, th: float, gx: float, gy: float, caps: np.ndarray, p: SimParams):
    """8-dim observation: rays / R, sin & cos of goal bearing relative to heading, scaled goal distance.

    Returns (obs, ray_distances). Bearing is computed with dot/cross products (no atan2):
    sin > 0 means the goal is clockwise (to the right on screen) of the heading.
    """
    rays = geometry.cast_rays(x, y, ray_dirs(th, p), caps, p.ray_range)
    hx = math.cos(th)
    hy = math.sin(th)
    ex = gx - x
    ey = gy - y
    dist = math.sqrt(ex * ex + ey * ey)
    if dist > 1e-9:
        ux = ex / dist
        uy = ey / dist
        cos_b = hx * ux + hy * uy
        sin_b = hx * uy - hy * ux
    else:
        cos_b, sin_b = 1.0, 0.0
    obs = np.empty(len(p.ray_angles) + 3)  # env appends the previous action if enabled
    k = len(p.ray_angles)
    obs[:k] = rays / p.ray_range
    obs[k] = sin_b
    obs[k + 1] = cos_b
    obs[k + 2] = min(dist / p.goal_dist_scale, 1.0)
    return obs, rays


def goal_distance(x: float, y: float, gx: float, gy: float) -> float:
    ex = gx - x
    ey = gy - y
    return math.sqrt(ex * ex + ey * ey)


__all__ = ["SimParams", "physics_step", "observe", "ray_dirs", "wrap_angle", "goal_distance", "geometry"]
