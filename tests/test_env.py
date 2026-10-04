import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mazebot import geometry as g
from mazebot import mapgen
from mazebot.env import MazeEnv


def open_map(start=(1.5, 3.0, 0.0), goal=(8.5, 3.0)):
    return mapgen.attach_field(mapgen.Map(10.0, 6.0, g.EMPTY.copy(), start, goal, "open"))


def rollout(env, action):
    total, steps = 0.0, 0
    while True:
        _, r, term, trunc, info = env.step(action)
        total += r
        steps += 1
        if term or trunc:
            return total, steps, term, trunc, info


def test_gymnasium_check_env():
    check_env(MazeEnv(), skip_render_check=True)


def test_obs_in_space_across_categories():
    env = MazeEnv(categories=mapgen.CATEGORIES)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(400):
        obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_driving_straight_to_goal_succeeds_with_positive_return():
    env = MazeEnv(maps=[open_map()])
    env.reset(seed=0)
    total, steps, term, _, info = rollout(env, [1.0, 1.0])
    assert term and info["success"]
    assert total > env.rew.goal + 0.8 * 7.0 - 0.5  # bonus + most of the 7-unit progress
    step_len = env.sim.v_max * env.sim.dt * env.sim.frame_skip
    assert steps == pytest.approx((7.0 - env.sim.goal_radius) / step_len, abs=2)


def test_driving_away_is_penalised_and_times_out():
    env = MazeEnv(maps=[open_map(start=(5.0, 3.0, math.pi))])
    env.reset(seed=0)
    total, _, _, trunc, info = rollout(env, [1.0, 1.0])
    assert trunc and not info["success"]
    assert total < -3.0
    assert info["contacts"] > 0  # ended up pushing against the border


def test_spinning_in_place_gets_no_progress():
    env = MazeEnv(maps=[open_map()])
    env.reset(seed=0)
    rewards = [env.step([1.0, -1.0])[1] for _ in range(30)]
    assert sum(rewards) == pytest.approx(-30 * env.rew.time, abs=0.05)


def test_seeded_reset_is_deterministic():
    a, b = MazeEnv(), MazeEnv()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.array_equal(oa, ob)
    for act in np.random.default_rng(0).uniform(-1, 1, (50, 2)):
        ra = a.step(act)
        rb = b.step(act)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]


def test_previous_action_memory_inputs():
    from mazebot.sim import SimParams

    env = MazeEnv(maps=[open_map()], sim=SimParams(prev_action_inputs=True))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,) and env.observation_space.shape == (10,)
    assert np.all(obs[8:] == 0.0)
    obs, *_ = env.step([0.5, -2.0])  # clipped to [-1, 1]
    assert obs[8:].tolist() == [0.5, -1.0]
    check_env(MazeEnv(sim=SimParams(prev_action_inputs=True)), skip_render_check=True)


def test_reverse_penalty_only_when_reversing_allowed_and_commanded():
    from mazebot.env import RewardParams
    from mazebot.sim import SimParams

    m = open_map(start=(5.0, 3.0, 0.0))
    env = MazeEnv(maps=[m], sim=SimParams(reverse_max=1.0), reward=RewardParams(reverse=0.1))
    env.reset(seed=0)
    x0 = env.x
    _, r_back, *_ = env.step([-1.0, -1.0])
    assert env.x < x0  # actually moved backwards
    env.reset(seed=0)
    _, r_spin, *_ = env.step([1.0, -1.0])
    # spinning in place: no reverse penalty; full reverse: 0.1 penalty (+ negative progress)
    assert r_spin == pytest.approx(-env.rew.time, abs=1e-3)
    assert r_back < r_spin - 0.1 + 1e-9
