"""SB3 VecEnv that runs several envs per subprocess (fewer IPC round trips than SubprocVecEnv).

With a cheap env (~0.15 ms/step) the pipe round trip dominates, so batching K envs per worker gives a
near-K-fold speed-up. Semantics match SubprocVecEnv: auto-reset, `terminal_observation` in info.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv


def _worker(remote, parent, fns_wrapper):
    parent.close()
    envs = [fn() for fn in fns_wrapper.var]
    obs_shape = envs[0].observation_space.shape
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                # Send stacked arrays; per-env info only at episode end (all SB3 and our callbacks read),
                # which keeps the pickled payload small.
                obs_out = np.empty((len(envs), *obs_shape), dtype=np.float32)
                rews = np.empty(len(envs), dtype=np.float32)
                dones = np.zeros(len(envs), dtype=bool)
                infos = {}
                for k, (env, a) in enumerate(zip(envs, data, strict=True)):
                    obs, r, term, trunc, info = env.step(a)
                    if term or trunc:
                        info["TimeLimit.truncated"] = trunc and not term
                        info["terminal_observation"] = obs
                        obs, _ = env.reset()
                        dones[k] = True
                        infos[k] = info
                    obs_out[k] = obs
                    rews[k] = r
                remote.send((obs_out, rews, dones, infos))
            elif cmd == "reset":
                remote.send([env.reset(seed=s)[0] for env, s in zip(envs, data, strict=True)])
            elif cmd == "env_method":
                name, args, kwargs, idx = data
                fns = [env.get_wrapper_attr(name) for i, env in enumerate(envs) if i in idx]
                remote.send([fn(*args, **kwargs) for fn in fns])
            elif cmd == "get_attr":
                name, idx = data
                remote.send([env.get_wrapper_attr(name) for i, env in enumerate(envs) if i in idx])
            elif cmd == "set_attr":
                name, value, idx = data
                for i, env in enumerate(envs):
                    if i in idx:
                        setattr(env.unwrapped, name, value)
                remote.send(None)
            elif cmd == "is_wrapped":
                from stable_baselines3.common.env_util import is_wrapped

                cls, idx = data
                remote.send([is_wrapped(env, cls) for i, env in enumerate(envs) if i in idx])
            elif cmd == "spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, n_workers: int):
        self.n = len(env_fns)
        n_workers = min(n_workers, self.n)
        self.splits = np.array_split(np.arange(self.n), n_workers)
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for idx in self.splits:
            parent, child = ctx.Pipe()
            fns = CloudpickleWrapper([env_fns[i] for i in idx])
            p = ctx.Process(target=_worker, args=(child, parent, fns), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.remotes[0].send(("spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        super().__init__(self.n, obs_space, act_space)
        self._seeds = [None] * self.n
        self.closed = False

    # -- core
    def step_async(self, actions):
        for remote, idx in zip(self.remotes, self.splits, strict=True):
            remote.send(("step", actions[idx]))

    def step_wait(self):
        parts = [remote.recv() for remote in self.remotes]
        obs = np.concatenate([p[0] for p in parts])
        rews = np.concatenate([p[1] for p in parts])
        dones = np.concatenate([p[2] for p in parts])
        infos = [p[3].get(k, {}) for p in parts for k in range(len(p[1]))]
        return obs, rews, dones, infos

    def reset(self):
        for remote, idx in zip(self.remotes, self.splits, strict=True):
            remote.send(("reset", [self._seeds[i] for i in idx]))
        obs = [o for remote in self.remotes for o in remote.recv()]
        self._seeds = [None] * self.n
        self._reset_seeds()
        return np.stack(obs)

    def seed(self, seed=None):
        if seed is None:
            return [None] * self.n
        self._seeds = [seed + i for i in range(self.n)]
        return self._seeds

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.procs:
            p.join()
        self.closed = True

    # -- attribute plumbing (per worker, filtered by global indices)
    def _route(self, indices):
        indices = list(self._get_indices(indices))
        for w, idx in enumerate(self.splits):
            local = [i - idx[0] for i in indices if idx[0] <= i <= idx[-1]]
            if local:
                yield w, local

    def get_attr(self, attr_name, indices=None):
        out = []
        for w, local in self._route(indices):
            self.remotes[w].send(("get_attr", (attr_name, local)))
            out.extend(self.remotes[w].recv())
        return out

    def set_attr(self, attr_name, value, indices=None):
        for w, local in self._route(indices):
            self.remotes[w].send(("set_attr", (attr_name, value, local)))
            self.remotes[w].recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        out = []
        for w, local in self._route(indices):
            self.remotes[w].send(("env_method", (method_name, method_args, method_kwargs, local)))
            out.extend(self.remotes[w].recv())
        return out

    def env_is_wrapped(self, wrapper_class, indices=None):
        out = []
        for w, local in self._route(indices):
            self.remotes[w].send(("is_wrapped", (wrapper_class, local)))
            out.extend(self.remotes[w].recv())
        return out
