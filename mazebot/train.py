"""PPO training with a map-category curriculum (Stable-Baselines3, CPU).

Usage: uv run python -m mazebot.train configs/mix_6x6.yaml [--steps N] [--name NAME]

Writes artifacts/runs/<name>/: config.yaml, tensorboard logs, checkpoints, weights.json (exported actor).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from . import mapgen
from .env import MazeEnv, RewardParams
from .policy import MLPPolicy
from .sim import SimParams

RUNS = Path("artifacts/runs")


def make_env(rank: int, seed: int, stage: dict, reward: dict):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        cats, w = zip(*stage.items(), strict=True)
        env = MazeEnv(categories=cats, weights=w, reward=RewardParams(**reward))
        env.reset(seed=seed * 1000 + rank)
        return Monitor(env, info_keywords=("success", "category"))

    return _init


def std_schedule_callback(start: float, end: float, steps: int):
    """Cap the policy's exploration std, annealed linearly from `start` to `end` over `steps`.

    The shipped policy is deterministic (the Gaussian mean). With large training noise PPO learns a
    mean that only works *because* of the noise (e.g. freezing in a dead-band that noise jiggles out
    of). Shrinking the noise forces the mean itself to drive well.
    """
    import math

    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    class StdSchedule(BaseCallback):
        def _cap(self):
            frac = min(1.0, self.num_timesteps / steps)
            cap = math.log(start + frac * (end - start))
            with torch.no_grad():
                self.model.policy.log_std.clamp_(max=cap)
            self.logger.record("train/std_cap", math.exp(cap))

        def _on_rollout_start(self):
            self._cap()

        def _on_step(self):
            return True

    return StdSchedule()


def curriculum_callback(stages, threshold, window, min_steps, max_steps, log_every=50_000):
    from stable_baselines3.common.callbacks import BaseCallback

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.stage = 0
            self.stage_start = 0
            self.recent = deque(maxlen=window)
            self.by_cat: dict[str, deque] = {}
            self.history = []
            self.last_log = 0
            self.stages = stages

        def _set_stage(self, k):
            self.stage = k
            self.stage_start = self.num_timesteps
            self.recent.clear()
            cats, w = zip(*stages[k].items(), strict=True)
            self.training_env.env_method("set_categories", cats, w)
            self.history.append({"stage": k, "step": self.num_timesteps, "mix": stages[k]})
            print(f"[curriculum] step {self.num_timesteps:,}: stage {k} {stages[k]}", flush=True)

        def _on_training_start(self):
            self.history.append({"stage": 0, "step": 0, "mix": stages[0]})

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
                if done:
                    s = float(info.get("success", False))
                    self.recent.append(s)
                    self.by_cat.setdefault(info["category"], deque(maxlen=500)).append(s)
            in_stage = self.num_timesteps - self.stage_start
            if self.stage < len(stages) - 1 and in_stage >= min_steps:
                rate = np.mean(self.recent) if len(self.recent) >= window else 0.0
                if rate >= threshold or in_stage >= max_steps:
                    self._set_stage(self.stage + 1)
            if self.num_timesteps - self.last_log >= log_every:
                self.last_log = self.num_timesteps
                self.logger.record("curriculum/stage", self.stage)
                if self.recent:
                    self.logger.record("curriculum/success", float(np.mean(self.recent)))
                for c, d in self.by_cat.items():
                    self.logger.record(f"success/{c}", float(np.mean(d)))
            return True

    return Curriculum()


def validation_callback(
    out: Path, curriculum, every: int, n: int, patience: int, min_delta: float, workers: int
):
    """Every `every` steps: export the actor, evaluate it on the validation maps (VAL_SEED, never the test
    set), keep the best weights, and stop once the final curriculum stage has plateaued.

    Score = mean success over the training categories (equal weight). Early stop when, in the final
    stage, `patience` consecutive validations fail to beat the best final-stage score by `min_delta`.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    from .evaluate import VAL_SEED, evaluate

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last = 0
            self.best = -1.0
            self.best_final = -1.0
            self.stale = 0
            self.history = []
            self.stopped_early = False

        def _on_step(self):
            if self.num_timesteps - self.last < every:
                return True
            self.last = self.num_timesteps
            final = curriculum.stage == len(curriculum_stages) - 1
            path = out / "val" / f"step_{self.num_timesteps:09d}.json"
            path.parent.mkdir(exist_ok=True)
            policy = MLPPolicy.from_sb3(self.model)
            policy.save(path, SimParams(), {"step": self.num_timesteps})
            report, _ = evaluate(str(path), n, mapgen.TRAIN_CATEGORIES, workers, seed=VAL_SEED)
            per_cat = {c: r["success"] for c, r in report["categories"].items()}
            score = float(np.mean(list(per_cat.values())))
            for c, v in per_cat.items():
                self.logger.record(f"val/{c}", v)
            self.logger.record("val/score", score)
            if score > self.best:
                self.best = score
                meta = {"step": self.num_timesteps, "val": per_cat}
                policy.save(out / "weights_best.json", SimParams(), meta)
            if final:
                if score > self.best_final + min_delta:
                    self.best_final = score
                    self.stale = 0
                else:
                    self.stale += 1
            rec = {"step": self.num_timesteps, "stage": curriculum.stage, "score": score, **per_cat}
            self.history.append(rec)
            with open(out / "validation.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            cats = " ".join(f"{c[:5]}={v:.2f}" for c, v in per_cat.items())
            stale = f" stale {self.stale}" if final else ""
            print(
                f"[val] step {self.num_timesteps:,} stage {curriculum.stage} score {score:.3f} ({cats}){stale}",  # noqa: E501
                flush=True,
            )
            if final and self.stale >= patience:
                print(f"[val] early stop: no gain > {min_delta} in {patience} validations", flush=True)
                self.stopped_early = True
                return False
            return True

    curriculum_stages = curriculum.stages
    return Validation()


def train(cfg: dict, name: str):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from .vec import BatchedSubprocVecEnv

    torch.set_num_threads(cfg.get("torch_threads", 2))
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cur = cfg["curriculum"]
    stages = cur["stages"]
    for st in stages:
        unknown = set(st) - set(mapgen.CATEGORIES)
        assert not unknown, f"unknown categories {unknown}"
    reward = cfg.get("reward", {})
    fns = [make_env(i, cfg["seed"], stages[0], reward) for i in range(cfg["n_envs"])]
    venv = BatchedSubprocVecEnv(fns, cfg.get("n_workers", 12))

    ppo = dict(cfg["ppo"])
    log_std_init = ppo.pop("log_std_init", 0.0)
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(
            net_arch=dict(pi=list(cfg["hidden"]), vf=list(cfg.get("value_hidden", [64, 64]))),
            activation_fn=torch.nn.Tanh,
            log_std_init=log_std_init,
        ),
        seed=cfg["seed"],
        device="cpu",
        tensorboard_log=str(out / "tb"),
        verbose=0,
        **ppo,
    )
    curriculum = curriculum_callback(
        stages, cur["threshold"], cur["window"], cur["min_steps"], cur["max_steps_per_stage"]
    )
    ckpt = CheckpointCallback(cfg.get("checkpoint_every", 2_000_000) // cfg["n_envs"], str(out / "ckpt"))
    v = cfg.get("validation", {})
    validation = validation_callback(
        out,
        curriculum,
        every=v.get("every", 1_000_000),
        n=v.get("n", 100),
        patience=v.get("patience", 4),
        min_delta=v.get("min_delta", 0.01),
        workers=v.get("workers", 14),
    )

    callbacks = [curriculum, validation, ckpt]
    if "std_schedule" in cfg:
        sc = cfg["std_schedule"]
        callbacks.append(std_schedule_callback(sc["start"], sc["end"], sc["steps"]))

    t0 = time.perf_counter()
    model.learn(
        cfg["total_steps"], callback=callbacks, tb_log_name="ppo", progress_bar=False
    )
    secs = time.perf_counter() - t0
    model.save(out / "model")
    venv.close()

    # weights.json = best validation checkpoint (what gets evaluated and shipped); weights_last.json = final
    last = MLPPolicy.from_sb3(model)
    meta = {
        "run": name,
        "steps_budget": cfg["total_steps"],
        "steps_done": model.num_timesteps,
        "stopped_early": validation.stopped_early,
        "best_val_score": validation.best,
        "train_seconds": secs,
        "curriculum": curriculum.history,
        "validation": validation.history,
    }
    last.save(out / "weights_last.json", SimParams(), meta)
    best = out / "weights_best.json"
    src = best if best.exists() else out / "weights_last.json"
    (out / "weights.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "train_summary.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done in {secs / 60:.1f} min ({model.num_timesteps / secs:,.0f} steps/s) -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.steps:
        cfg["total_steps"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    name = args.name or f"{Path(args.config).stem}_s{cfg['seed']}"
    train(cfg, name)


if __name__ == "__main__":
    main()
