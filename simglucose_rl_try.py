import os
import pickle
import pandas as pd
import numpy as np
from collections import deque
import math

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import Wrapper
from gymnasium import RewardWrapper

from simglucose.simulation.scenario import CustomScenario
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# 1) simglucose 환경 등록
register(
    id="simglucose-adol2-v0",
    entry_point="simglucose.envs:T1DSimGymnaisumEnv",
    max_episode_steps=200,
    kwargs={"patient_name": "adolescent#002"},
)

class MultiHistoryWrapper(Wrapper):
    def __init__(self, env, history_length: int = 12):
        super().__init__(env)
        self.history_length = history_length
        self.bg_buf      = deque(maxlen=history_length)
        self.meal_buf    = deque(maxlen=history_length)
        self.insulin_buf = deque(maxlen=history_length)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        init_bg   = info["bg"]
        init_meal = info["meal"]
        init_ins  = 0.0
        for _ in range(self.history_length):
            self.bg_buf.append(init_bg)
            self.meal_buf.append(init_meal)
            self.insulin_buf.append(init_ins)
        info["hist_bg"]      = np.array(self.bg_buf)
        info["hist_meal"]    = np.array(self.meal_buf)
        info["hist_insulin"] = np.array(self.insulin_buf)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ins = float(np.array(action).ravel()[0]) if isinstance(action, (np.ndarray, list, tuple)) else float(action)
        self.bg_buf.append(info["bg"])
        self.meal_buf.append(info["meal"])
        self.insulin_buf.append(ins)
        info["hist_bg"]      = np.array(self.bg_buf)
        info["hist_meal"]    = np.array(self.meal_buf)
        info["hist_insulin"] = np.array(self.insulin_buf)
        return obs, reward, terminated, truncated, info

class LGBMRewardWrapper(Wrapper):
    def __init__(self, env, model_path: str):
        super().__init__(env)
        base_dir = os.path.dirname(os.path.realpath(__file__))
        abs_model_path = os.path.join(base_dir, model_path)
        with open(abs_model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.transformer = self.pipeline.named_steps["transform"]
        self.regressor   = self.pipeline.named_steps["regressor"]
        self.bg_gap = 5

    def gaussian_reward(self, pred_bg: float, target: float = 125.0, sigma: float = 25.0) -> float:
        return math.exp(-0.5 * ((pred_bg - target) / sigma) ** 2)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # 시간 -> sin/cos
        t = info["time"]
        minutes = t.hour * 60 + t.minute
        sin_t = np.sin(2 * np.pi * minutes / 1440)
        cos_t = np.cos(2 * np.pi * minutes / 1440)

        # 히스토리
        hist_bg      = info["hist_bg"]
        current_bg   = float(info["bg"])
        bg_lags      = np.concatenate([[current_bg], hist_bg])
        bg_diff      = np.diff(bg_lags)
        insulin_hist = info["hist_insulin"]
        carbs_hist   = info["hist_meal"]

        # 피처 생성
        p_num = "p02"
        feat_arr = np.concatenate([
            [p_num], [self.bg_gap], [sin_t], [cos_t],
            bg_lags, bg_diff,
            insulin_hist, carbs_hist
        ]).reshape(1, -1)

        input_cols = list(self.transformer.feature_names_in_)
        feat_df    = pd.DataFrame(feat_arr, columns=input_cols)
        for c in feat_df.columns:
            if c != "p_num":
                feat_df[c] = feat_df[c].astype(float)

        # 예측
        Xt = self.transformer.transform(feat_df)
        if isinstance(Xt, pd.DataFrame):
            Xt = Xt.astype(float)
        pred_bg = float(self.regressor.predict(Xt)[0])

        # 보상 계산: gaussian + severe penalty + delta 보너스
        reward = self.gaussian_reward(pred_bg)
        # 극단적 hypo/hype 페널티
        if pred_bg < 70:
            reward -= (1.0 + 0.05 * (70 - pred_bg))
        elif pred_bg > 180:
            reward -= (1.0 + 0.02 * (pred_bg - 180))

        # 상승 보너스
        delta = pred_bg - current_bg
        if current_bg < 80 and delta > 0:
            reward += 0.2 * (delta / 10.0)

        new_reward = np.clip(reward, -1.0, 1.0)
        return obs, new_reward, terminated, truncated, info


def make_env():
    base = gym.make("simglucose-adol2-v0")
    hist = MultiHistoryWrapper(base, history_length=12)
    mon  = Monitor(hist)
    rew  = LGBMRewardWrapper(mon, model_path="lgbm_model.pkl")
    return rew


def main():
    os.makedirs("logs", exist_ok=True)

    vec_env = DummyVecEnv([make_env for _ in range(4)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./logs/"
    )

    eval_env = Monitor(make_env())
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True
    )

    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("ppo_simglucose_hist_tree_adol2")

    mean_reward, std_reward = evaluate_policy(
        model, vec_env, n_eval_episodes=10, render=False
    )
    print(f"[Evaluation] Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    single = make_env()
    obs, info = single.reset(seed=123)
    bg_history = []
    time_steps = []
    max_steps = single.env.env.env.spec.max_episode_steps

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = single.step(action)
        bg_history.append(info["bg"])
        time_steps.append(t)
        if terminated or truncated:
            print(f"Demo finished at step {t+1}")
            break

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, bg_history, marker='o')
    plt.xlabel("Time step")
    plt.ylabel("Blood Glucose (mg/dL)")
    plt.title("Blood Glucose Control under Learned PPO Policy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bg_control_plot.png")
    plt.close()
    print("Saved blood glucose control plot to bg_control_plot.png")

    single.close()

if __name__ == "__main__":
    main()
