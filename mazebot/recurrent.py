"""Recurrent (GRU) actor: training via sb3-contrib RecurrentPPO, export + NumPy inference.

sb3-contrib's RecurrentPPO is built around nn.LSTM (states = (h, c)). `GRUAsLSTM` presents a GRU through
the same interface (it carries h and returns a zero "cell" state), so `install_gru` can swap it into a
RecurrentPPO policy. Actor = obs -> GRU(H) -> MLP head (tanh) -> linear -> clip; that is what ships.

PyTorch GRU (gate order r, z, n):
    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h          # z ~ 1: keep memory, z ~ 0: overwrite
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sim import SimParams

WEIGHTS_VERSION = 1


# --------------------------------------------------------------------------- training (torch)


def _torch():
    import torch
    from torch import nn

    return torch, nn


def make_gru_as_lstm(input_size: int, hidden_size: int):
    torch, nn = _torch()

    class GRUAsLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers=1)
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = 1

        def forward(self, x, states):
            h, c = states
            out, h_new = self.gru(x, h.contiguous())
            return out, (h_new, torch.zeros_like(c))

    return GRUAsLSTM()


def install_gru(model) -> None:
    """Replace the LSTMs of a RecurrentPPO policy with GRUs of the same size (and rebuild the optimizer)."""
    pol = model.policy
    for name in ("lstm_actor", "lstm_critic"):
        lstm = getattr(pol, name)
        if lstm is not None:
            setattr(pol, name, make_gru_as_lstm(lstm.input_size, lstm.hidden_size).to(pol.device))
    pol.optimizer = pol.optimizer_class(pol.parameters(), lr=model.lr_schedule(1), **pol.optimizer_kwargs)
    pol._process_sequence = fast_process_sequence  # instance attribute shadows the staticmethod


def fast_process_sequence(features, lstm_states, episode_starts, lstm):
    """Drop-in for RecurrentActorCriticPolicy._process_sequence.

    sb3-contrib falls back to a per-timestep Python loop whenever any episode starts inside the batch.
    In training, sequences are already split at episode boundaries, so starts only ever occur at a
    sequence's first step: then we zero those sequences' initial state and run the fused GRU over the
    whole sequence in one call. Anything else (starts mid-sequence) uses the original loop.
    """
    torch, _ = _torch()
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

    n_seq = lstm_states[0].shape[1]
    seq = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)  # (T, n_seq, F)
    starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)  # (T, n_seq)
    if starts.shape[0] > 1 and torch.any(starts[1:] != 0.0):
        return RecurrentActorCriticPolicy._process_sequence(features, lstm_states, episode_starts, lstm)
    keep = (1.0 - starts[0]).view(1, n_seq, 1)
    out, states = lstm(seq, (keep * lstm_states[0], keep * lstm_states[1]))
    return torch.flatten(out.transpose(0, 1), start_dim=0, end_dim=1), states


# --------------------------------------------------------------------------- shipped actor (NumPy)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRUPolicy:
    """Stateful NumPy actor. Call reset() at the start of every episode."""

    def __init__(self, gru: dict, W: list[np.ndarray], b: list[np.ndarray], sim: SimParams | None = None):
        self.W_ih, self.W_hh = np.asarray(gru["W_ih"]), np.asarray(gru["W_hh"])
        self.b_ih, self.b_hh = np.asarray(gru["b_ih"]), np.asarray(gru["b_hh"])
        self.W, self.b = W, b
        self.sim = sim
        self.H = self.W_hh.shape[1]
        self.reset()

    @property
    def arch(self) -> dict:
        return {
            "inputs": self.W_ih.shape[1],
            "gru": self.H,
            "head": [w.shape[0] for w in self.W[:-1]],
            "outputs": self.W[-1].shape[0],
        }

    def reset(self):
        self.h = np.zeros(self.H)

    def step(self, obs, h):
        """One GRU step + head. Returns (action, new_h, details) — details feed the network panel."""
        x = np.asarray(obs, dtype=np.float64)
        H = self.H
        gi = self.W_ih @ x + self.b_ih
        gh = self.W_hh @ h + self.b_hh
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H : 2 * H] + gh[H : 2 * H])
        n = np.tanh(gi[2 * H :] + r * gh[2 * H :])
        h_new = (1.0 - z) * n + z * h
        a = h_new
        acts = [h_new]
        for k, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
            a = w @ a + b
            if k < len(self.W) - 1:
                a = np.tanh(a)
            acts.append(a)
        return np.clip(a, -1.0, 1.0), h_new, {"r": r, "z": z, "n": n, "layers": acts}

    def __call__(self, obs):
        action, self.h, _ = self.step(obs, self.h)
        return action

    @classmethod
    def from_sb3(cls, model) -> GRUPolicy:
        torch, nn = _torch()
        pol = model.policy
        g = pol.lstm_actor.gru

        def np_(t):
            return t.detach().cpu().double().numpy()

        gru = {
            "W_ih": np_(g.weight_ih_l0),
            "W_hh": np_(g.weight_hh_l0),
            "b_ih": np_(g.bias_ih_l0),
            "b_hh": np_(g.bias_hh_l0),
        }
        layers = [m for m in pol.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [pol.action_net]
        return cls(gru, [np_(m.weight) for m in layers], [np_(m.bias) for m in layers])

    def to_json(self, sim: SimParams, meta: dict | None = None) -> dict:
        from .policy import obs_spec

        return {
            "version": WEIGHTS_VERSION,
            "type": "gru",
            "arch": self.arch,
            "hidden_activation": "tanh",
            "output": "clip",
            "obs_spec": obs_spec(sim),
            "act_spec": ["left wheel", "right wheel"],
            "sim_params": sim.to_dict(),
            "gru": {
                k: v.tolist()
                for k, v in (
                    ("W_ih", self.W_ih),
                    ("W_hh", self.W_hh),
                    ("b_ih", self.b_ih),
                    ("b_hh", self.b_hh),
                )
            },
            "W": [w.tolist() for w in self.W],
            "b": [b.tolist() for b in self.b],
            "meta": meta or {},
        }

    def save(self, path: str | Path, sim: SimParams, meta: dict | None = None):
        Path(path).write_text(json.dumps(self.to_json(sim, meta)), encoding="utf-8", newline="\n")

    @classmethod
    def from_json(cls, d: dict) -> GRUPolicy:
        return cls(
            d["gru"],
            [np.array(w) for w in d["W"]],
            [np.array(b) for b in d["b"]],
            SimParams.from_dict(d.get("sim_params")),
        )
