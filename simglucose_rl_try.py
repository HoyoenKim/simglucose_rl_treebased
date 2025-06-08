import os
import pickle
from collections import deque
from typing import Any, Deque, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# =============================================================================
# Constants & Hyperparameters
# =============================================================================
ENV_ID = "simglucose-adol2-v0"
HISTORY_LENGTH = 12
NUM_ENVS = 4
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
MODEL_PATH = "lgbm_model.pkl"
LOG_DIR = "./logs"
BEST_MODEL_DIR = "./logs/best_model"


# =============================================================================
# Utility: Continuous Reward Function
# =============================================================================
def continuous_reward(pred_bg: float, target: float = 125.0, sigma: float = 25.0) -> float:
    """
    Compute reward based on deviation from target BG.

    - |delta| <= sigma: +10
    - delta < -sigma: linear interp from -10 at -2σ to +10 at -σ
    - delta > +sigma: linear interp from +10 at +σ to -5 at +2σ
    """
    delta = pred_bg - target

    # within one sigma
    if abs(delta) <= sigma:
        return 10.0

    # below target
    if delta < -sigma:
        if delta <= -2 * sigma:
            return -10.0
        frac = (delta + 2 * sigma) / sigma
        return -10.0 + frac * 20.0

    # above target
    if delta > sigma:
        if delta >= 2 * sigma:
            return -5.0
        frac = (delta - sigma) / sigma
        return 10.0 - frac * 15.0

    return 0.0


# =============================================================================
# Callback: Log training episode rewards
# =============================================================================
class TrainRewardCallback(BaseCallback):
    """
    Record episode rewards during training and save a plot at end.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.timesteps: list[int] = []
        self.rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(ep_info["r"])
        return True

    def _on_training_end(self) -> None:
        plt.figure()
        plt.plot(self.timesteps, self.rewards)
        plt.xlabel("Timestep")
        plt.ylabel("Episode Reward")
        plt.title("Training: Timestep vs Reward")
        plt.tight_layout()
        plt.savefig("train_reward.png")
        plt.close()


# =============================================================================
# Wrapper: History of BG, meal, insulin
# =============================================================================
class MultiHistoryWrapper(Wrapper):
    """Maintain fixed-length history for BG, meal, insulin"""
    def __init__(self, env: gym.Env, history_length: int = HISTORY_LENGTH):
        super().__init__(env)
        self.history_length = history_length
        self.bg_buf: Deque[float] = deque(maxlen=history_length)
        self.meal_buf: Deque[float] = deque(maxlen=history_length)
        self.insulin_buf: Deque[float] = deque(maxlen=history_length)

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        init_bg, init_meal = info["bg"], info["meal"]
        for _ in range(self.history_length):
            self.bg_buf.append(init_bg)
            self.meal_buf.append(init_meal)
            self.insulin_buf.append(0.0)
        info.update(
            hist_bg=np.array(self.bg_buf),
            hist_meal=np.array(self.meal_buf),
            hist_insulin=np.array(self.insulin_buf),
        )
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, done, trunc, info = self.env.step(action)
        ins = float(np.array(action).ravel()[0]) if hasattr(action, "__iter__") else float(action)
        self.bg_buf.append(info["bg"])
        self.meal_buf.append(info["meal"])
        self.insulin_buf.append(ins)
        info.update(
            hist_bg=np.array(self.bg_buf),
            hist_meal=np.array(self.meal_buf),
            hist_insulin=np.array(self.insulin_buf),
        )
        return obs, reward, done, trunc, info


# =============================================================================
# Wrapper: LightGBM pipeline reward
# =============================================================================
class LGBMRewardWrapper(Wrapper):
    """Compute reward via pretrained Transformer+Regressor pipeline"""
    def __init__(self, env: gym.Env, model_path: str = MODEL_PATH):
        super().__init__(env)
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        self.transformer = pipeline.named_steps["transform"]
        self.regressor = pipeline.named_steps["regressor"]

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, dict]:
        obs, _, done, trunc, info = self.env.step(action)
        t = info["time"]
        minutes = t.hour * 60 + t.minute
        sin_t, cos_t = (
            np.sin(2 * np.pi * minutes / 1440),
            np.cos(2 * np.pi * minutes / 1440),
        )
        hist_bg = info["hist_bg"]
        current_bg = float(info["bg"])
        bg_lags = np.concatenate([[current_bg], hist_bg])
        feat = np.concatenate([
            ["p02"],
            [5],
            [sin_t, cos_t],
            bg_lags,
            np.diff(bg_lags),
            info["hist_insulin"],
            info["hist_meal"],
        ])
        df = pd.DataFrame(
            feat.reshape(1, -1),
            columns=self.transformer.feature_names_in_,
        )
        for col in df.columns:
            if col != "p_num":
                df[col] = df[col].astype(float)
        Xt = self.transformer.transform(df)
        Xt = Xt.astype(float) if hasattr(Xt, "astype") else Xt
        pred = float(self.regressor.predict(Xt)[0])
        reward = continuous_reward(pred, target=110.0, sigma=10.0)
        return obs, reward, done, trunc, info


# =============================================================================
# Environment Registration & Factory
# =============================================================================
def register_env() -> None:
    register(
        id=ENV_ID,
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=200,
        kwargs={"patient_name": "adolescent#002"},
    )


def make_env() -> gym.Env:
    env = gym.make(ENV_ID)
    env = MultiHistoryWrapper(env)
    env = Monitor(env)
    env = LGBMRewardWrapper(env)
    return env


# =============================================================================
# Metrics & Evaluation
# =============================================================================
def compute_metrics_ppo(model: PPO, n_episodes: int = 20) -> Dict[str, float]:
    tir_count = tir_total = 0
    lbgi_list: list[float] = []
    hbgi_list: list[float] = []
    for ep in range(n_episodes):
        env = make_env()
        obs, info = env.reset(seed=ep)
        lbgi_vals: list[float] = []
        hbgi_vals: list[float] = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            bg = info["bg"]
            fbg = 1.509 * ((np.log(bg)) ** 1.084 - 5.381)
            rbg = 10 * (fbg ** 2)
            if fbg < 0:
                lbgi_vals.append(rbg)
            else:
                hbgi_vals.append(rbg)
            tir_total += 1
            if 70 <= bg <= 180:
                tir_count += 1
        env.close()
        lbgi_list.append(np.mean(lbgi_vals) if lbgi_vals else 0.0)
        hbgi_list.append(np.mean(hbgi_vals) if hbgi_vals else 0.0)
    return {
        "Time-in-Range (%)": 100 * tir_count / tir_total if tir_total else 0.0,
        "LBGI (mean)": float(np.mean(lbgi_list)),
        "HBGI (mean)": float(np.mean(hbgi_list)),
    }


def evaluate_and_plot_ppo(
    model: PPO, n_eval: int = 20, ep_len: int = 288
) -> Dict[str, float]:
    # 1) SB3 evaluate
    eval_env = DummyVecEnv([make_env])
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval, deterministic=True
    )
    print(f"Eval {n_eval} eps: {mean_r:.2f} ± {std_r:.2f}")

    # 2) Custom metrics
    metrics = compute_metrics_ppo(model, n_episodes=n_eval)
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")

    # 3) Trajectory plot (single episode)
    env = make_env()
    obs, info = env.reset(seed=0)
    bg_traj: list[float] = []
    ins_traj: list[float] = []
    for _ in range(ep_len):
        action, _ = model.predict(obs, deterministic=True)
        ins_traj.append(float(np.array(action).ravel()[0]))
        bg_traj.append(info["bg"])
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    env.close()

    plt.figure()
    plt.plot(bg_traj, marker="o")
    plt.title("BG Trajectory")
    plt.xlabel("Timestep (5 min)")
    plt.ylabel("BG (mg/dL)")
    plt.tight_layout()
    plt.savefig("eval_result_bg.png")
    plt.close()

    plt.figure()
    plt.plot(ins_traj, marker="o")
    plt.title("Insulin Trajectory")
    plt.xlabel("Timestep (5 min)")
    plt.ylabel("Insulin (IU)")
    plt.tight_layout()
    plt.savefig("eval_result_ins.png")
    plt.close()

    return metrics


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    register_env()

    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # Callbacks
    train_cb = TrainRewardCallback()
    eval_cb = EvalCallback(
        Monitor(make_env()),
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )
    prog_cb = ProgressBarCallback()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[train_cb, eval_cb, prog_cb],
    )
    model.save("ppo_simglucose_hist_tree_adol2")

    # Post-training evaluation
    evaluate_and_plot_ppo(model)


if __name__ == "__main__":
    main()
