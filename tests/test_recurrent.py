import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from mazebot.env import MazeEnv  # noqa: E402
from mazebot.policy import export_actor, load_policy  # noqa: E402
from mazebot.recurrent import GRUPolicy, install_gru  # noqa: E402
from mazebot.sim import SimParams  # noqa: E402

SIM = SimParams(prev_action_inputs=True)


def make_model():
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO(
        "MlpLstmPolicy",
        MazeEnv(sim=SIM),
        policy_kwargs=dict(
            lstm_hidden_size=8,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(pi=[4], vf=[16]),
            activation_fn=torch.nn.Tanh,
        ),
        n_steps=16,
        batch_size=16,
        device="cpu",
        seed=0,
    )
    install_gru(model)
    with torch.no_grad():  # make the memory matter: larger random weights
        for p in model.policy.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    return model


def test_gru_swapped_in_and_trains_one_update():
    model = make_model()
    assert type(model.policy.lstm_actor).__name__ == "GRUAsLSTM"
    model.learn(32)  # one rollout + update runs end to end with the GRU


def test_numpy_gru_matches_sb3_over_a_sequence_with_reset(tmp_path):
    model = make_model()
    pol = export_actor(model)
    assert isinstance(pol, GRUPolicy)
    assert pol.arch == {"inputs": 10, "gru": 8, "head": [4], "outputs": 2}
    obs_seq = np.random.default_rng(0).uniform(-1, 1, (40, 10)).astype(np.float32)
    starts = np.zeros(40, dtype=bool)
    starts[0] = starts[25] = True  # new episode at t=25: memory must reset
    state = None
    pol.reset()
    for t in range(40):
        theirs, state = model.predict(
            obs_seq[t : t + 1], state=state, episode_start=starts[t : t + 1], deterministic=True
        )
        if starts[t]:
            pol.reset()
        ours = pol(obs_seq[t].astype(np.float64))
        assert np.abs(ours - theirs[0]).max() < 1e-5, t
    # memory matters: same observation, different history -> different action
    pol.reset()
    a_fresh = pol(obs_seq[0])
    for o in obs_seq[1:10]:
        pol(o)
    assert np.abs(pol(obs_seq[0]) - a_fresh).max() > 1e-3

    path = tmp_path / "gru.json"
    pol.save(path, SIM)
    loaded = load_policy(path)
    assert isinstance(loaded, GRUPolicy) and loaded.sim.prev_action_inputs
    loaded.reset()
    pol.reset()
    for o in obs_seq[:15]:
        assert np.array_equal(loaded(o), pol(o))


def test_fast_sequence_path_matches_sb3_loop():
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

    from mazebot.recurrent import fast_process_sequence, make_gru_as_lstm

    torch.manual_seed(0)
    gru = make_gru_as_lstm(10, 8)
    n_seq, T = 6, 20
    feats = torch.randn(n_seq * T, 10)
    h0 = (torch.randn(1, n_seq, 8), torch.zeros(1, n_seq, 8))
    starts = torch.zeros(n_seq, T)
    starts[[0, 3], 0] = 1.0  # some sequences begin a new episode (training case)
    flat = starts.reshape(-1)
    ref, ref_h = RecurrentActorCriticPolicy._process_sequence(feats, h0, flat, gru)
    out, h = fast_process_sequence(feats, h0, flat, gru)
    assert torch.allclose(out, ref, atol=1e-6) and torch.allclose(h[0], ref_h[0], atol=1e-6)
    starts[2, 7] = 1.0  # a start mid-sequence must fall back to the exact loop
    flat = starts.reshape(-1)
    ref, _ = RecurrentActorCriticPolicy._process_sequence(feats, h0, flat, gru)
    out, _ = fast_process_sequence(feats, h0, flat, gru)
    assert torch.allclose(out, ref, atol=1e-6)
