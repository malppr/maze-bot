"""The exported policy: a plain MLP (tanh hidden layers, linear output clipped to [-1, 1]).

`MLPPolicy` is what ships to the browser (`weights.json`) and what `evaluate.py` runs, so reported
results are for exactly the exported network, not the PyTorch model it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sim import SimParams

WEIGHTS_VERSION = 1


@dataclass
class MLPPolicy:
    W: list[np.ndarray]  # W[k]: (out, in)
    b: list[np.ndarray]
    sim: SimParams | None = None

    @property
    def arch(self) -> list[int]:
        return [self.W[0].shape[1]] + [w.shape[0] for w in self.W]

    def forward(self, obs: np.ndarray, return_activations: bool = False):
        """obs (..., in) -> actions (..., 2). Optionally all layer activations (for the network panel)."""
        h = np.asarray(obs, dtype=np.float64)
        acts = [h]
        for k, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
            h = h @ w.T + b
            if k < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        out = np.clip(h, -1.0, 1.0)
        return (out, acts) if return_activations else out

    __call__ = forward

    # ------------------------------------------------------------- io

    @classmethod
    def from_sb3(cls, model) -> MLPPolicy:
        """Extract the actor from an SB3 PPO model (mlp_extractor.policy_net + action_net)."""
        import torch.nn as nn

        layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
        layers.append(model.policy.action_net)
        W = [layer.weight.detach().cpu().double().numpy() for layer in layers]
        b = [layer.bias.detach().cpu().double().numpy() for layer in layers]
        return cls(W, b)

    def to_json(self, sim: SimParams, meta: dict | None = None) -> dict:
        return {
            "version": WEIGHTS_VERSION,
            "arch": self.arch,
            "hidden_activation": "tanh",
            "output": "clip",
            "obs_spec": [f"ray{i + 1}/R" for i in range(len(sim.ray_angles))]
            + ["sin(goal bearing)", "cos(goal bearing)", "goal distance / scale"]
            + (["previous left wheel", "previous right wheel"] if sim.prev_action_inputs else []),
            "act_spec": ["left wheel", "right wheel"],
            "sim_params": sim.to_dict(),
            "W": [w.tolist() for w in self.W],
            "b": [b.tolist() for b in self.b],
            "meta": meta or {},
        }

    def save(self, path: str | Path, sim: SimParams, meta: dict | None = None):
        Path(path).write_text(json.dumps(self.to_json(sim, meta)), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> MLPPolicy:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pol = cls([np.array(w) for w in d["W"]], [np.array(b) for b in d["b"]])
        pol.sim = SimParams.from_dict(d.get("sim_params"))  # sensors/kinematics it was trained with
        return pol
