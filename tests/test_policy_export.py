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
