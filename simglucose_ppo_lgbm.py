# -*- coding: utf-8 -*-
from __future__ import annotations

# ───────────────────────────────────────────
# Standard library imports
# ───────────────────────────────────────────
import argparse
import math
import os
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

# ───────────────────────────────────────────
# Third‑party imports
# ───────────────────────────────────────────
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from torch.distributions import Beta
from torch.nn import functional as F

# ───────────────────────────────────────────
# Constants & Hyper‑parameters
# ───────────────────────────────────────────
# Environment
ENV_ID: str = "simglucose-adol2-v0"
PATIENT_NAME: str = "adolescent#002"  # SimGlucose synthetic patient profile
HISTORY_LENGTH: int = 12            # 1h history (12×5min)
EPISODE_STEPS: int = 288            # 24h × (60/5)=288

# Training / evaluation
NUM_ENVS: int = int(os.environ.get("SIMGLU_ENVS", 4))
TOTAL_TIMESTEPS: int = int(os.environ.get("SIMGLU_STEPS", 2_304_000))  # 8000 eps (override for smoke)
EVAL_FREQ: int = 144_000            # evaluate every 500 episodes
N_EVAL_EPISODES: int = 5
SEED: int = 42
WARM_UP_STEPS : int = 20_000

# ── Bug-fix toggle ────────────────────────────────────────────────
# SIMGLU_FIX=1 (default): corrected LightGBM feature ordering + predicted BG
# actually fed to the policy observation.  SIMGLU_FIX=0: original (buggy)
# behaviour, kept so both can be A/B-compared at an equal training budget.
FIX: bool = os.environ.get("SIMGLU_FIX", "1") != "0"
TAG: str = os.environ.get("SIMGLU_TAG", "")          # keep A/B artefacts apart
_suffix = f"_{TAG}" if TAG else ""

# Paths
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
BEST_MODEL_DIR = LOG_DIR / f"best_model{_suffix}"
MODEL_CKPT = BASE_DIR / f"ppo_simglucose_hist_tree_adol2{_suffix}.zip"
VEC_NORM_STATS = BASE_DIR / f"vec_normalize_stats{_suffix}.pkl"
LGBM_MODEL = BASE_DIR / "lgbm_model.pkl"  # transformer+ LightGBM pipeline

# Misc.
BG_TARGET: float = 125.0  # mg/dL
BG_SIGMA: float = 25.0    # Reward width parameter
# Reward shaping actually used by LGBMRewardWrapper. Kept explicit to remove the
# old doc/code drift (call site used 140/30 while constants claimed 125/25).
REWARD_TARGET: float = 140.0  # mg/dL
REWARD_SIGMA: float = 30.0

# ───────────────────────────────────────────
# 1.Reward utility
# ───────────────────────────────────────────

def continuous_reward(pred_bg: float, *, target: float = BG_TARGET, sigma: float = BG_SIGMA) -> float:
    """Smooth Gaussian‑like reward centred at *target*.

    The curve peaks at +0.648 above baseline when BG==target, equals 0 at
    |delta| == sigma, and decays outside ±sigma with additional asymmetrical
    scaling to penalise severe hypoglycaemia more strongly.
    """
    delta = pred_bg - target
    expo = math.exp(-0.5 * (delta / sigma) ** 2 + 0.5) - 1  # shift so baseline=0

    # Tail scaling (asymmetric): hypoglycaemia>> hyperglycaemia
    if delta < -2.0 * sigma:
        return expo * 10.0
    if delta > 4.0 * sigma:
        return expo * 5.0
    if delta > 2.0 * sigma:
        return expo * 2.0
    return expo + 1  # ensure reward ≥0 inside ±sigma

# ───────────────────────────────────────────
# 2.Gym wrappers
# ───────────────────────────────────────────

class MultiHistoryWrapper(gym.Wrapper):
    """Augment *info* with 1‑hour history (BG, meal, insulin)."""
    def __init__(self, env: gym.Env, history_len: int = HISTORY_LENGTH):
        super().__init__(env)
        self._hist_len = history_len
        self._bg: Deque[float] = deque(maxlen=history_len)
        self._meal: Deque[float] = deque(maxlen=history_len)
        self._ins: Deque[float] = deque(maxlen=history_len)

    # Gymnasium API ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        init_bg, init_meal = info["bg"], info["meal"]
        for _ in range(self._hist_len):
            self._bg.append(init_bg)
            self._meal.append(init_meal)
            self._ins.append(0.0)
        info.update(hist_bg=np.array(self._bg), hist_meal=np.array(self._meal), hist_insulin=np.array(self._ins))
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        insulin_scalar = float(np.array(action).ravel()[0])
        self._bg.append(info["bg"])
        self._meal.append(info["meal"])
        self._ins.append(insulin_scalar)
        info.update(hist_bg=np.array(self._bg), hist_meal=np.array(self._meal), hist_insulin=np.array(self._ins))
        return obs, rew, term, trunc, info


class LGBMRewardWrapper(gym.Wrapper):
    """Replace SimGlucose's native reward with ML‑predicted risk‑aware reward."""
    def __init__(self, env: gym.Env, *, model_path: Path = LGBM_MODEL, lamb_ins: float = 0.2, lamb_risk: float = 0.1):
        super().__init__(env)
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        self._transform = pipeline.named_steps["transform"]
        self._regressor = pipeline.named_steps["regressor"]
        self._lamda_ins = lamb_ins
        self._lamda_risk = lamb_risk
        self._prev_risk = 0.0

    # — internal helpers — --------------------------------------------------
    @staticmethod
    def _risk(bg: float) -> float:
        f_bg = 1.509 * ((math.log(max(bg, 1e-3))) ** 1.084 - 5.381)
        return 10.0 * (f_bg ** 2)

    # Gymnasium API ---------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_risk = self._risk(float(info.get("bg", 100.0)))
        info.update(sin_t=0.0, cos_t=0.0, pred_bg60=float(info.get("bg", 100.0)))
        return obs, info

    def _build_feature_row(self, current_bg, sin_t, cos_t, info):
        """Assemble the 1-row DataFrame in the exact column order the saved
        LightGBM pipeline expects. FIX corrects the time ordering: the model
        wants newest-first (bg_0_lag = now ... bg_60_lag = 60 min ago), but the
        original code fed oldest-first with a duplicated 'now'."""
        cols = self._transform.feature_names_in_
        if FIX:
            bg_hist = np.asarray(info["hist_bg"], dtype=float) / 18.0182   # oldest->newest (12)
            bg_rev = bg_hist[::-1]                                          # newest->oldest
            bg_lag = np.concatenate([bg_rev, [bg_rev[-1]]])                 # 13 (bg_60 dropped by pipe)
            bg_diff = bg_lag[:-1] - bg_lag[1:]                             # 12 (newer - older)
            ins_rev = np.asarray(info["hist_insulin"], dtype=float)[::-1]   # newest->oldest (12)
            carb_rev = np.asarray(info["hist_meal"], dtype=float)[::-1]     # newest->oldest (12)
            values = np.concatenate([["p02"], [5], [sin_t, cos_t],
                                     bg_lag, bg_diff, ins_rev, carb_rev])
        else:
            hist_bg = np.asarray(info["hist_bg"], dtype=float) / 18.0182
            bg_lags = np.concatenate([hist_bg, [current_bg / 18.0182]])
            values = np.concatenate([["p02"], [5], [sin_t, cos_t],
                                     bg_lags, np.diff(bg_lags),
                                     np.asarray(info["hist_insulin"], dtype=float),
                                     np.asarray(info["hist_meal"], dtype=float)])
        df = pd.DataFrame(values.reshape(1, -1), columns=cols)
        for col in df.columns:
            if col != "p_num":
                df[col] = df[col].astype(float)
        return df

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        t: datetime = info["time"]
        minutes = t.hour * 60 + t.minute
        sin_t, cos_t = np.sin(2 * np.pi * minutes / 1440.0), np.cos(2 * np.pi * minutes / 1440.0)

        current_bg = float(info["bg"])
        # Build 60‑minute feature window (mmol→mg conversion inside)
        hist_bg = info["hist_bg"] / 18.0182
        df = self._build_feature_row(current_bg, sin_t, cos_t, info)
        Xt = self._transform.transform(df).astype(float)
        pred_bg = float(self._regressor.predict(Xt)[0]) * 18.0182 # mmol→mg

        # simple 2‑h linear extrapolation fallback
        bg_all = np.concatenate([[current_bg], info["hist_bg"]])
        t_past = np.arange(5, 5 * (len(hist_bg) + 1), 5)
        slope, intercept = np.polyfit(np.concatenate([[0], t_past]), bg_all, 1)
        bg_future = max(1.0, slope * 120 + intercept)

        bg_for_reward = bg_future if abs(pred_bg - current_bg) >= 50 else pred_bg
        base = continuous_reward(bg_for_reward, target=REWARD_TARGET, sigma=REWARD_SIGMA)

        curr_risk = self._risk(bg_for_reward)
        reward = base + self._lamda_risk * (self._prev_risk - curr_risk) - self._lamda_ins * float(np.array(action).ravel()[0])
        reward = float(np.clip(reward, -5.0, 5.0))
        self._prev_risk = curr_risk

        info.update(sin_t=sin_t, cos_t=cos_t, pred_bg60=bg_for_reward)
        return obs, reward, term, trunc, info


class SafetyFilter(gym.ActionWrapper):
    """Rule‑based *fail‑safe* overlay on the agent's insulin suggestions."""
    def __init__(self, env: gym.Env, *, low: float = 80.0, high: float = 250.0, min_bolus: float = 0.01):
        super().__init__(env)
        self._low, self._high, self._min = low, high, min_bolus
        self._prev_bg: Optional[float] = None
        self._last_bg: Optional[float] = None

    # — helper — -----------------------------------------------------------
    def _safe_action(self, raw_a: float) -> float:
        bg = self._last_bg
        if bg is None:
            return raw_a  # 1st step
        d_bg = bg - (self._prev_bg or bg)
        if bg < self._low or d_bg < 0:
            return 0.0
        if bg > self._high:
            return max(raw_a, self._min)
        return raw_a

    # Gymnasium API ---------------------------------------------------------
    def action(self, action):
        a_scalar = float(np.array(action).ravel()[0])
        safe = self._safe_action(a_scalar)
        if isinstance(self.action_space, Box):
            lo = float(np.ravel(self.action_space.low)[0])
            hi = float(np.ravel(self.action_space.high)[0])
            safe = float(np.clip(safe, lo, hi))
        return np.asarray([safe], dtype=self.action_space.dtype)  # ensure shape (1,)

    def step(self, action):
        self._prev_bg = self._last_bg
        obs, rew, term, trunc, info = self.env.step(self.action(action))
        self._last_bg = info["bg"]
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_bg = info["bg"]
        return obs, info


class FeatureObsWrapper(gym.Wrapper):
    """Convert scalar BG obs →  40‑dim feature vector for PPO."""
    def __init__(self, env: gym.Env, *, hist_len: int = HISTORY_LENGTH):
        super().__init__(env)
        self._hist_len = hist_len
        feat_dim = 1 + 3 * hist_len + 2 + 1  # BG + (BG/meal/ins)*H + sin/cos + pred
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32)

    # helper ---------------------------------------------------------------
    def _build(self, obs, info):
        obs = np.asarray(obs).flatten()
        h_bg = np.asarray(info.get("hist_bg", np.zeros(self._hist_len)), dtype=np.float32)
        h_ins = np.asarray(info.get("hist_insulin", np.zeros(self._hist_len)), dtype=np.float32)
        h_meal = np.asarray(info.get("hist_meal", np.zeros(self._hist_len)), dtype=np.float32)
        sincos = np.array([info.get("sin_t", 0.0), info.get("cos_t", 0.0)], dtype=np.float32)
        # FIX: read the key the reward wrapper actually writes ("pred_bg60").
        # With FIX off, keep the original (broken) "pred_bg30" -> obs[-1] fallback.
        pred_key = "pred_bg60" if FIX else "pred_bg30"
        pred = np.array([info.get(pred_key, obs[-1])], dtype=np.float32)
        return np.concatenate([obs, h_bg, h_ins, h_meal, sincos, pred])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._build(obs, info), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._build(obs, info), rew, term, trunc, info


# ───────────────────────────────────────────
# 3.Custom Beta‑policy for continuous dosing
# ───────────────────────────────────────────

class CustomBetaPolicy(ActorCriticPolicy):
    """PPO policy head producing (α, β) parameters of a Beta distribution."""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if not (isinstance(action_space, Box) and action_space.shape == (1,)):
            raise ValueError("Action space must be continuous scalar (Box(1,)).")
        self.param_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 2)

    def _get_dist(self, latent_pi: th.Tensor):
        raw = self.param_net(latent_pi)
        alpha = F.softplus(raw[:, 0]) + 1.0
        beta = F.softplus(raw[:, 1]) + 1.0
        return Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1))

# ───────────────────────────────────────────
# 4.Environment factory & registration
# ───────────────────────────────────────────

def _register_env() -> None:
    """Idempotent registration of the SimGlucose Gymnasium env."""
    if ENV_ID not in gym.envs.registry:
        register(
            id=ENV_ID,
            entry_point="simglucose.envs:T1DSimGymnaisumEnv",
            max_episode_steps=EPISODE_STEPS,
            kwargs={"patient_name": PATIENT_NAME},
        )


def env_factory() -> gym.Env:
    """Return **one** fully wrapped environment instance."""
    _register_env()  # idempotent; required inside SubprocVecEnv workers (spawn/forkserver)
    env = gym.make(ENV_ID)
    env = MultiHistoryWrapper(env)
    env = LGBMRewardWrapper(env)
    env = SafetyFilter(env)
    env = FeatureObsWrapper(env)
    env = Monitor(env)
    return env

# ───────────────────────────────────────────
# 5.Auxiliary callbacks
# ───────────────────────────────────────────

class TrainPlotCallback(BaseCallback):
    """Collect per‑episode rewards & dump *train_reward.png* at the end."""
    def __init__(self):
        super().__init__()
        self.x: List[int] = []
        self.y: List[float] = []

    def _on_step(self) -> bool:  # noqa: D401 (Stable‑Baselines3 naming)
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self.x.append(self.num_timesteps)
                self.y.append(ep["r"])
        return True

    def _on_training_end(self) -> None:  # type: ignore[override]
        plt.figure()
        plt.plot(self.x, self.y)
        plt.xlabel("Timestep")
        plt.ylabel("Episode reward")
        plt.title("Training curve")
        plt.tight_layout()
        plt.savefig(LOG_DIR / "train_reward.png")
        plt.close()

# ───────────────────────────────────────────
# 6.Training / evaluation helpers
# ───────────────────────────────────────────

def _build_vec_env(n_envs: int, *, load_stats: bool = False) -> VecNormalize:
    """Create *DummyVecEnv → VecNormalize* (optionally restore running‑stats)."""
    use_subproc = os.environ.get("SIMGLU_SUBPROC", "0") == "1" and n_envs > 1
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv  # SubprocVecEnv parallelises the slow env across processes
    venv = vec_cls([env_factory for _ in range(n_envs)])
    vec = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=5.0)
    if load_stats and VEC_NORM_STATS.exists():
        vec = VecNormalize.load(str(VEC_NORM_STATS), venv)
    return vec


def train_rl_agent() -> None:
    """(Re‑)train a PPO agent; checkpoint + plots are written to disk."""
    LOG_DIR.mkdir(exist_ok=True)
    BEST_MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # 1.Build / restore VecNormalize env
    vec_env = _build_vec_env(NUM_ENVS, load_stats=VEC_NORM_STATS.exists())
    vec_env.seed(SEED)

    # 2.Load existing checkpoint if present
    reset_num_timesteps = True
    if MODEL_CKPT.exists():
        print("[INFO] Continuing from existing checkpoint…")
        model = PPO.load(MODEL_CKPT, env=vec_env)
        if not VEC_NORM_STATS.exists():
            obs = vec_env.reset()
            print("[INFO] Warming up vector env")
            for _ in range(WARM_UP_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = vec_env.step(action)
            print("[INFO] Warming up vector env end... Start training")
            reset_num_timesteps = False
    else:
        model = PPO(
            policy=CustomBetaPolicy,
            env=vec_env,
            verbose=1,
            tensorboard_log=str(LOG_DIR),
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=EPISODE_STEPS,  # learn across one entire simulated day
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    # 3.Set up callbacks
    eval_vec = _build_vec_env(1, load_stats=True)
    eval_vec.training = False

    callbacks = [
        TrainPlotCallback(),
        ProgressBarCallback(),
        EvalCallback(
            eval_vec,
            best_model_save_path=str(BEST_MODEL_DIR),
            log_path=str(LOG_DIR),
            eval_freq=EVAL_FREQ // NUM_ENVS,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
        ),
    ]

    # 4.Learn
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=reset_num_timesteps, callback=callbacks)

    # 5.Persist artefacts
    print("[INFO] Saving checkpoint →", MODEL_CKPT)
    model.save(MODEL_CKPT)
    vec_env.save(str(VEC_NORM_STATS))


def _bg_insulin_plots(bg: List[float], ins: List[float]) -> None:
    """Save *eval_result_bg.png* and *eval_result_ins.png*."""
    plt.figure()
    plt.plot(bg, marker="o")
    plt.title("BG trajectory (evaluation)")
    plt.xlabel("Timestep (5min)")
    plt.ylabel("BG (mg/dL)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_result_bg.png")
    plt.close()

    plt.figure()
    plt.plot(ins, marker="o")
    plt.title("Insulin trajectory (evaluation)")
    plt.xlabel("Timestep (5min)")
    plt.ylabel("Insulin (IU)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_result_ins.png")
    plt.close()


def evaluate_agent(*, episodes: int = 20) -> None:
    """Evaluate the saved model and print metrics.

    Fixes vs. the original: (1) observations are normalised with the *training*
    VecNormalize statistics before being fed to the policy, and (2) each episode
    uses a *distinct* seed, so the across-episode spread is real (it was
    identically seeded -> +-0.00 before)."""
    if not MODEL_CKPT.exists():
        raise FileNotFoundError("No trained model found. Run `train` first.")

    vec_env = _build_vec_env(1, load_stats=True)
    vec_env.training = False
    vec_env.norm_reward = False
    if not VEC_NORM_STATS.exists():
        print("[WARN] vec_normalize_stats missing -> obs NOT normalised; numbers unreliable.")
    model = PPO.load(MODEL_CKPT, env=vec_env)

    # SB3 helper for mean episode reward (runs through the normalised vec-env)
    mean_r, std_r = evaluate_policy(model, vec_env, n_eval_episodes=episodes, deterministic=True)
    print(f"Mean episode reward over {episodes}eps: {mean_r:.2f}+-{std_r:.2f}")

    # Custom BG safety metrics (per-episode TIR so we can report a real spread)
    per_ep_tir, lbgi_vals, hbgi_vals = [], [], []
    tbr54 = total = 0
    for ep in range(episodes):
        env = env_factory()
        obs, info = env.reset(seed=SEED + ep)           # distinct seed per episode
        done = False
        ep_in = ep_tot = 0
        while not done:
            nobs = vec_env.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(nobs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            bg = float(info["bg"])
            fbg = 1.509 * ((np.log(max(bg, 1e-3))) ** 1.084 - 5.381)
            rbg = 10 * fbg ** 2
            (lbgi_vals if fbg < 0 else hbgi_vals).append(rbg)
            ep_tot += 1
            total += 1
            if 70 <= bg <= 180:
                ep_in += 1
            if bg < 54:
                tbr54 += 1
        env.close()
        per_ep_tir.append(100.0 * ep_in / max(ep_tot, 1))

    print(f"Time-in-Range: {np.mean(per_ep_tir):.2f}% +- {np.std(per_ep_tir):.2f}  (n={episodes} seeds)")
    print(f"LBGI (mean):   {np.mean(lbgi_vals) if lbgi_vals else 0.0:.2f}")
    print(f"HBGI (mean):   {np.mean(hbgi_vals) if hbgi_vals else 0.0:.2f}")
    print(f"Time-below-54: {100.0 * tbr54 / max(total, 1):.2f}%  (severe hypo)")

    # Single-episode trace for visual sanity check (also normalised)
    viz_env = env_factory()
    bg_traj, ins_traj = [], []
    obs, info = viz_env.reset(seed=SEED)
    for _ in range(EPISODE_STEPS):
        nobs = vec_env.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(nobs, deterministic=True)
        ins_traj.append(float(np.array(action).ravel()[0]))
        bg_traj.append(info["bg"])
        obs, _, term, trunc, info = viz_env.step(action)
        if term or trunc:
            break
    viz_env.close()
    _bg_insulin_plots(bg_traj, ins_traj)

# ───────────────────────────────────────────
# 7.Entry‑point
# ───────────────────────────────────────────

def main() -> None:
    _register_env()
    parser = argparse.ArgumentParser(description="SimGlucose RL pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train or continue training the agent")
    train_p.set_defaults(func=lambda _: train_rl_agent())

    eval_p = sub.add_parser("eval", help="Evaluate the saved agent")
    eval_p.add_argument("--episodes", type=int, default=20, help="#episodes for evaluation")
    eval_p.set_defaults(func=lambda args: evaluate_agent(episodes=args.episodes))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
