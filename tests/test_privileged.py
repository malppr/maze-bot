import numpy as np
import pytest

torch = pytest.importorskip("torch")  # train group only

from gymnasium.utils.env_checker import check_env  # noqa: E402

from mazebot.env import N_CRITIC_EXTRAS, MazeEnv  # noqa: E402
from mazebot.policy import MLPPolicy  # noqa: E402
from mazebot.sim import SimParams  # noqa: E402

SIM = SimParams(prev_action_inputs=True)


def make_model():
    from stable_baselines3 import PPO

    from mazebot.privileged import PrivilegedCriticPolicy

    env = MazeEnv(sim=SIM, critic_extras=True)
    return PPO(
        PrivilegedCriticPolicy,
        env,
        policy_kwargs=dict(
            net_arch=dict(pi=[12, 12], vf=[32, 32]), activation_fn=torch.nn.Tanh, n_actor_obs=SIM.obs_dim
        ),
        device="cpu",
        seed=0,
    )


def test_env_extras_shape_range_and_checker():
    env = MazeEnv(sim=SIM, critic_extras=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (SIM.obs_dim + N_CRITIC_EXTRAS,)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, *_ = env.step(rng.uniform(-1, 1, 2))
        assert env.observation_space.contains(obs)
    check_env(MazeEnv(sim=SIM, critic_extras=True), skip_render_check=True)


def test_actor_ignores_privileged_inputs_but_critic_uses_them():
    model = make_model()
    obs = np.random.default_rng(1).uniform(-1, 1, (64, SIM.obs_dim + N_CRITIC_EXTRAS)).astype(np.float32)
    other = obs.copy()
    other[:, SIM.obs_dim :] = np.random.default_rng(2).uniform(-1, 1, (64, N_CRITIC_EXTRAS))
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = model.predict(other, deterministic=True)
    assert np.array_equal(a1, a2)  # actor output identical whatever the extras say
    with torch.no_grad():
        v1 = model.policy.predict_values(torch.as_tensor(obs))
        v2 = model.policy.predict_values(torch.as_tensor(other))
    assert not torch.allclose(v1, v2)  # critic does read them


def test_export_is_the_plain_actor_and_runs_without_extras():
    model = make_model()
    pol = MLPPolicy.from_sb3(model)
    assert pol.arch == [SIM.obs_dim, 12, 12, 2]
    obs = np.random.default_rng(3).uniform(-1, 1, (32, SIM.obs_dim + N_CRITIC_EXTRAS)).astype(np.float32)
    theirs, _ = model.predict(obs, deterministic=True)
    ours = pol(obs[:, : SIM.obs_dim].astype(np.float64))  # exported net only gets the robot's own inputs
    assert np.abs(ours - theirs).max() < 1e-5
    env = MazeEnv(sim=SIM)  # evaluation env: no extras
    o, _ = env.reset(seed=0)
    env.step(pol(o))


def test_path_direction_extra_points_along_shortest_path():
    from mazebot import geometry as g
    from mazebot import mapgen

    # wall between A and B with a gap at the bottom: the shortest path from A first heads down (+y)
    walls = g.capsules([[5, 0, 5, 4.5, 0.1]])
    m = mapgen.attach_field(mapgen.Map(10.0, 6.0, walls, (2.0, 2.0, 0.0), (8.0, 2.0), "obstacles"))
    env = MazeEnv(maps=[m], sim=SIM, critic_extras=True, critic_path_dir=True)
    obs, _ = env.reset(seed=0, options={"map": m})
    assert obs.shape == (SIM.obs_dim + N_CRITIC_EXTRAS + 2,)
    path_sin, path_cos = obs[-2:]
    assert path_sin > 0.3  # heading +x, route bends towards +y = clockwise = positive sin
    assert abs(np.hypot(path_sin, path_cos) - 1) < 1e-5
