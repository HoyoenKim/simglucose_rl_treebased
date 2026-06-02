# -*- coding: utf-8 -*-
"""SAC baseline ported from baseline.ipynb (risk-delta reward + optional scale-down
safety filter; NO LightGBM), with a corrected multi-seed evaluation.

Toggles (env vars):
  SAC_STEPS     training timesteps (default 100000, matching the notebook)
  SAC_NOFILTER  =1 to drop the SACSafetyFilter (the "SAC (baseline)" row)
  SAC_TAG       suffix to keep checkpoints apart (e.g. filt / nofilt)

Usage:  python sac_simglucose.py train ;  python sac_simglucose.py eval --episodes 20
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import Wrapper, ActionWrapper
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "simglucose-adol2-v0"
PATIENT_NAME = "adolescent#002"
EPISODE_STEPS = 288
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
TOTAL_TIMESTEPS = int(os.environ.get("SAC_STEPS", 100_000))
USE_FILTER = os.environ.get("SAC_NOFILTER", "0") != "1"
TAG = os.environ.get("SAC_TAG", "")
_suf = f"_{TAG}" if TAG else ""
MODEL_CKPT = BASE_DIR / f"sac_simglucose_adol2{_suf}.zip"

# Reward / filter hyperparameters (from baseline.ipynb)
SAC_SCALE_RISK, SAC_LAMBDA_INS = 0.1, 0.2
SAC_HYPO1, SAC_HYPO2, SAC_HYPER = 70, 60, 250
SAC_SAFETY_SCALE, SAC_SAFETY_CUTOFF = 0.5, 80


def _register_env() -> None:
    if ENV_ID not in gym.envs.registry:
        register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv",
                 max_episode_steps=EPISODE_STEPS, kwargs={"patient_name": PATIENT_NAME})


class SACGymCompat(Wrapper):
    """Ensure a Gymnasium 5-tuple regardless of the underlying gym/gymnasium API."""
    def step(self, action: Any):
        result = self.env.step(action)
        if len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, done, False, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}


class SACRiskDeltaPen(Wrapper):
    """reward = scale_risk*(prev_risk - risk) - lambda*|action| + hypo/hyper/TIR shaping."""
    def __init__(self, env, scale_risk=SAC_SCALE_RISK, lambda_ins=SAC_LAMBDA_INS,
                 hypo1=SAC_HYPO1, hypo2=SAC_HYPO2, hyper=SAC_HYPER):
        super().__init__(env)
        self.scale_risk, self.lambda_ins = scale_risk, lambda_ins
        self.h1, self.h2, self.H = hypo1, hypo2, hyper
        self.prev_risk = 0.0

    @staticmethod
    def _risk(bg: float) -> float:
        f = 1.509 * ((np.log(max(bg, 1e-3))) ** 1.084 - 5.381)
        return 10 * (f ** 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_risk = self._risk(info.get("bg", 100.0))
        return obs, info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        bg = info.get("bg", 100.0)
        current_risk = self._risk(bg)
        reward = self.scale_risk * (self.prev_risk - current_risk)
        reward -= self.lambda_ins * float(np.abs(np.asarray(action).item()))
        if bg < self.h2:
            reward -= 10.0
        elif bg < self.h1:
            reward -= 5.0
        elif bg > self.H:
            reward -= 2.0
        else:
            reward += 10.0
        self.prev_risk = current_risk
        return obs, float(reward), done, truncated, info


class SACSafetyFilter(ActionWrapper):
    """Scale the action down when BG < cutoff (the notebook's SAC filter)."""
    def __init__(self, env, scale=SAC_SAFETY_SCALE, cutoff=SAC_SAFETY_CUTOFF):
        super().__init__(env)
        self.scale, self.cutoff = scale, cutoff
        self._last_info: dict = {}

    def action(self, action):
        bg = self._last_info.get("bg", None)
        if bg is not None and bg < self.cutoff:
            return np.asarray(action) * self.scale
        return action

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._last_info = info
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        return obs, info


def sac_make_env() -> gym.Env:
    _register_env()
    env = gym.make(ENV_ID)
    env = SACGymCompat(env)
    env = SACRiskDeltaPen(env)
    if USE_FILTER:
        env = SACSafetyFilter(env)
    env = Monitor(env)
    return env


def train() -> None:
    env = DummyVecEnv([sac_make_env])
    model = SAC("MlpPolicy", env, learning_rate=3e-4, batch_size=256, buffer_size=500_000,
                tau=0.005, gamma=0.99, ent_coef="auto_0.1", verbose=1)
    print(f"[SAC] filter={USE_FILTER} steps={TOTAL_TIMESTEPS}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(MODEL_CKPT)
    print("[SAC] saved ->", MODEL_CKPT)


def evaluate(episodes: int = 20) -> None:
    if not MODEL_CKPT.exists():
        raise FileNotFoundError(f"No SAC model at {MODEL_CKPT}; run train first.")
    model = SAC.load(MODEL_CKPT)
    per_ep_tir, lbgi_vals, hbgi_vals = [], [], []
    tbr54 = total = 0
    for ep in range(episodes):
        env = sac_make_env()
        obs, info = env.reset(seed=SEED + ep)
        done = False
        ep_in = ep_tot = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            bg = float(info["bg"])
            f = 1.509 * ((np.log(max(bg, 1e-3))) ** 1.084 - 5.381)
            (lbgi_vals if f < 0 else hbgi_vals).append(10 * f ** 2)
            ep_tot += 1
            total += 1
            if 70 <= bg <= 180:
                ep_in += 1
            if bg < 54:
                tbr54 += 1
        env.close()
        per_ep_tir.append(100.0 * ep_in / max(ep_tot, 1))
    print(f"[SAC filter={USE_FILTER}] eval n={episodes} seeds")
    print(f"Time-in-Range: {np.mean(per_ep_tir):.2f}% +- {np.std(per_ep_tir):.2f}")
    print(f"LBGI (mean):   {np.mean(lbgi_vals) if lbgi_vals else 0.0:.2f}")
    print(f"HBGI (mean):   {np.mean(hbgi_vals) if hbgi_vals else 0.0:.2f}")
    print(f"Time-below-54: {100.0 * tbr54 / max(total, 1):.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAC baseline (risk-delta, no LightGBM)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(f=lambda a: train())
    e = sub.add_parser("eval")
    e.add_argument("--episodes", type=int, default=20)
    e.set_defaults(f=lambda a: evaluate(a.episodes))
    args = p.parse_args()
    args.f(args)
