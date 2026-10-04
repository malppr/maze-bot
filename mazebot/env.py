"""Gymnasium environment: Wheely drives from A to B on sampled maps."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from . import geometry, mapgen
from .sim import SimParams, goal_distance, observe, physics_step

N_CRITIC_EXTRAS = 10


@dataclass(frozen=True)
class RewardParams:
    progress: float = 1.0  # per world unit of geodesic progress
    contact: float = 0.05  # per policy step in wall contact (0.2 taught it to freeze; see PLAN.md)
    time: float = 0.01  # per policy step
    reverse: float = 0.0  # per step x commanded backward speed in [0, 1] (needs reverse_max > 0)
    goal: float = 10.0  # on reaching B
    timeout_factor: float = 3.0  # time limit = factor * (geodesic A->B / v_max) + slack
    timeout_slack: float = 10.0  # seconds


class MazeEnv(gym.Env):
    """Observation: 8 floats (see sim.observe). Action: (uL, uR) in [-1, 1]^2.

    `categories` and `weights` choose which map generator each episode uses (curriculum hook:
    `set_categories`). Pass `maps` to cycle through fixed maps instead (evaluation / tests).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        categories: Sequence[str] = mapgen.TRAIN_CATEGORIES,
        weights: Sequence[float] | None = None,
        maps: Sequence[mapgen.Map] | None = None,
        sim: SimParams | None = None,
        reward: RewardParams | None = None,
        map_fn: Callable[[np.random.Generator], mapgen.Map] | None = None,
        render_mode: str | None = None,
        critic_extras: bool = False,
    ):
        self.sim = sim or SimParams()
        self.rew = reward or RewardParams()
        self.render_mode = render_mode
        # Training-only privileged inputs for the critic, appended after the actor's inputs. The actor never
        # sees them (see mazebot/privileged.py); evaluation and export use critic_extras=False.
        self.critic_extras = critic_extras
        self.n_actor_obs = self.sim.obs_dim
        self.set_categories(categories, weights)
        self.maps = list(maps) if maps is not None else None
        self.map_fn = map_fn
        self._map_idx = 0
        n_obs = self.sim.obs_dim + (N_CRITIC_EXTRAS if critic_extras else 0)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.map: mapgen.Map | None = None

    def set_categories(self, categories: Sequence[str], weights: Sequence[float] | None = None):
        self.categories = tuple(categories)
        w = np.ones(len(self.categories)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()

    # ------------------------------------------------------------------ gym API

    def _next_map(self) -> mapgen.Map:
        if self.maps is not None:
            m = self.maps[self._map_idx % len(self.maps)]
            self._map_idx += 1
            if m.field is None:
                mapgen.attach_field(m)
            return m
        if self.map_fn is not None:
            return self.map_fn(self.np_random)
        cat = self.categories[int(self.np_random.choice(len(self.categories), p=self.weights))]
        return mapgen.sample_map(self.np_random, cat)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "map" in options:
            self.map = options["map"]
            if self.map.field is None:
                mapgen.attach_field(self.map)
        else:
            self.map = self._next_map()
        m = self.map
        self.caps = m.caps
        self.x, self.y, self.th = m.start
        self.gx, self.gy = m.goal
        self.geo = m.geo(self.x, self.y)
        self.geo0 = self.geo
        policy_dt = self.sim.dt * self.sim.frame_skip
        limit_s = self.rew.timeout_factor * self.geo0 / self.sim.v_max + self.rew.timeout_slack
        self.max_steps = math.ceil(limit_s / policy_dt)
        self.steps = 0
        self.contacts = 0
        self.path_len = 0.0
        self.prev_action = np.zeros(2)
        obs, self.rays = observe(self.x, self.y, self.th, self.gx, self.gy, self.caps, self.sim)
        obs = self._with_extras(self._with_memory(obs))
        return obs.astype(np.float32), self._info(success=False)

    def step(self, action):
        ul, ur = (float(np.clip(a, -1.0, 1.0)) for a in action)
        p = self.sim
        contact = False
        success = False
        x0, y0 = self.x, self.y
        for _ in range(p.frame_skip):
            px, py = self.x, self.y
            self.x, self.y, self.th, c = physics_step(self.x, self.y, self.th, ul, ur, self.caps, p)
            self.path_len += math.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)
            contact = contact or c
            if goal_distance(self.x, self.y, self.gx, self.gy) < p.goal_radius:
                success = True
                break
        self.steps += 1
        self.contacts += int(contact)

        geo = self.map.geo(self.x, self.y)
        if not math.isfinite(geo):  # pushed somewhere the field doesn't reach: no progress signal
            geo = self.geo
        r = self.rew
        reward = r.progress * (self.geo - geo) - r.time - (r.contact if contact else 0.0)
        reward -= r.reverse * max(0.0, -0.5 * (ul + ur))
        if success:
            reward += r.goal
        self.geo = geo
        self.last_move = math.sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2)

        self.prev_action = np.array([ul, ur])
        obs, self.rays = observe(self.x, self.y, self.th, self.gx, self.gy, self.caps, p)
        obs = self._with_extras(self._with_memory(obs))
        truncated = (not success) and self.steps >= self.max_steps
        return obs.astype(np.float32), float(reward), success, truncated, self._info(success, contact)

    def _with_extras(self, obs: np.ndarray) -> np.ndarray:
        """Privileged critic inputs: shortest-path distance to B, 8 all-round rays, fraction of time left.

        (Deliberately no shortest-path *direction*: the critic should judge how good a situation is,
        not be handed the route.)
        """
        if not self.critic_extras:
            return obs
        th = self.th
        dirs = np.array([[math.cos(th + k * math.pi / 4), math.sin(th + k * math.pi / 4)] for k in range(8)])
        rays = geometry.cast_rays(self.x, self.y, dirs, self.caps, self.sim.ray_range) / self.sim.ray_range
        extras = np.empty(N_CRITIC_EXTRAS)
        extras[0] = min(self.geo / 20.0, 1.0) if math.isfinite(self.geo) else 1.0
        extras[1:9] = rays
        extras[9] = 1.0 - self.steps / self.max_steps
        return np.concatenate([obs, extras])

    def _with_memory(self, obs: np.ndarray) -> np.ndarray:
        if self.sim.prev_action_inputs:
            return np.concatenate([obs, self.prev_action])
        return obs

    def _info(self, success: bool, contact: bool = False) -> dict:
        return {
            "success": success,
            "contact": contact,
            "geo": self.geo,
            "geo0": self.geo0,
            "steps": self.steps,
            "contacts": self.contacts,
            "category": self.map.category,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            from .render import render_rgb

            return render_rgb(self)
        return None
