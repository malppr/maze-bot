import numpy as np
import pytest

from mazebot.policy import MLPPolicy
from mazebot.sim import SimParams

torch = pytest.importorskip("torch")  # train group only; CI runs without it


def test_numpy_actor_matches_sb3_deterministic(tmp_path):
    from stable_baselines3 import PPO

    from mazebot.env import MazeEnv

    model = PPO(
        "MlpPolicy",
        MazeEnv(),
        policy_kwargs=dict(net_arch=dict(pi=[6, 6], vf=[16]), activation_fn=torch.nn.Tanh),
        device="cpu",
        seed=0,
    )
    # perturb weights so outputs leave the clip range sometimes
    with torch.no_grad():
        for p in model.policy.parameters():
            p.add_(torch.randn_like(p) * 0.8)
    pol = MLPPolicy.from_sb3(model)
    assert pol.arch == [8, 6, 6, 2]
    assert sum(w.size + b.size for w, b in zip(pol.W, pol.b, strict=True)) == 110

    obs = np.random.default_rng(0).uniform(-1, 1, (256, 8)).astype(np.float32)
    ours = pol(obs.astype(np.float64))
    theirs, _ = model.predict(obs, deterministic=True)
    assert np.abs(ours - theirs).max() < 1e-5

    path = tmp_path / "w.json"
    pol.save(path, SimParams())
    assert np.array_equal(MLPPolicy.load(path)(obs), ours)


def test_custom_ray_layout_roundtrips_and_drives_eval(tmp_path):
    from mazebot import mapgen
    from mazebot.env import MazeEnv
    from mazebot.evaluate import make_policy, run_episode

    sim = SimParams.from_dict({"ray_angles_deg": [0, -30, 30, -60, 60, -90, 90]})
    env = MazeEnv(sim=sim)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)
    rng = np.random.default_rng(0)
    W = [rng.normal(size=(12, 10)), rng.normal(size=(12, 12)), rng.normal(size=(2, 12))]
    b = [np.zeros(12), np.zeros(12), np.zeros(2)]
    path = tmp_path / "w.json"
    MLPPolicy(W, b).save(path, sim)
    pol = make_policy(str(path))
    assert pol.arch == [10, 12, 12, 2]
    assert pol.sim.ray_angles_deg == sim.ray_angles_deg
    m = mapgen.sample_map(np.random.default_rng(1), "obstacles")
    run_episode(pol, m)  # would fail with a shape error if eval used the default 5-ray sensors
