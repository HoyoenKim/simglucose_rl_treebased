import os
import pickle
import pandas as pd
import numpy as np
from collections import deque

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

def clipped_reward(pred_bg: float) -> float:
    """
    pred_bg: 1시간 후 예측 혈당 값
    반환값: 혈당 범위별로 정의된 보상 (clipped 형태)
    """
    if 80 <= pred_bg <= 140:
        # 정상 구간: 최대 +1
        return +1.0
    elif 70 <= pred_bg < 80:
        # 약간 저혈당: 약간의 마이너스
        return -0.5
    elif 140 < pred_bg <= 180:
        # 약간 고혈당: 약간의 마이너스
        return -0.5
    elif pred_bg < 70:
        # 심각한 저혈당: 큰 음수
        return -2.0 - 0.05 * (70 - pred_bg)
    else:  # pred_bg > 180
        # 심각한 고혈당: 중간 음수
        return -2.0 - 0.02 * (pred_bg - 180)

# 1) simglucose 환경 등록
register(
    id="simglucose-adol2-v0",
    entry_point="simglucose.envs:T1DSimGymnaisumEnv",
    max_episode_steps=200,
    kwargs={
        "patient_name": "adolescent#002",
    },
)

class MultiHistoryWrapper(Wrapper):
    """
    - 과거 history_length 스텝만큼의 혈당(bg), 식사(meal), 인슐린(insulin) 이력을 저장하여
      info["hist_bg"], info["hist_meal"], info["hist_insulin"]에 numpy 배열 형태로 추가
    """
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
        init_ins  = 0.0  # reset 직후엔 action이 없으므로 0으로 초기화
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
        # action이 배열 형태일 수 있어, 스칼라 값으로 변환
        if isinstance(action, (np.ndarray, list, tuple)):
            ins = float(np.array(action).ravel()[0])
        else:
            ins = float(action)
        self.bg_buf.append(info["bg"])
        self.meal_buf.append(info["meal"])
        self.insulin_buf.append(ins)
        info["hist_bg"]      = np.array(self.bg_buf)
        info["hist_meal"]    = np.array(self.meal_buf)
        info["hist_insulin"] = np.array(self.insulin_buf)
        return obs, reward, terminated, truncated, info

class LGBMRewardWrapper(Wrapper):
    """
    - LightGBM으로 1스텝 후 혈당(pred_bg)을 예측하여 clipped_reward(pred_bg) 반환
    """
    def __init__(self, env, model_path: str):
        super().__init__(env)
        # 스크립트 파일이 위치한 폴더를 기준으로 pkl 파일 절대 경로 생성
        base_dir = os.path.dirname(os.path.realpath(__file__))
        abs_model_path = os.path.join(base_dir, model_path)
        with open(abs_model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.transformer = self.pipeline.named_steps["transform"]
        self.regressor   = self.pipeline.named_steps["regressor"]
        self.bg_gap = 5

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # 1) 시간 표현: 시각 → sin, cos
        t = info["time"]
        minutes = t.hour * 60 + t.minute
        sin_t = np.sin(2 * np.pi * minutes / 1440)
        cos_t = np.cos(2 * np.pi * minutes / 1440)

        # 2) 과거 히스토리와 현재 혈당
        hist_bg      = info["hist_bg"]       # shape=(history_length,)
        current_bg   = float(info["bg"])
        bg_lags      = np.concatenate([[current_bg], hist_bg])  # (history_length+1,)
        bg_diff      = np.diff(bg_lags)                       # (history_length,)
        insulin_hist = info["hist_insulin"]  # (history_length,)
        carbs_hist   = info["hist_meal"]     # (history_length,)

        # 3) 피처 배열 생성
        #    p_num, bg_gap, sin_t, cos_t, 현재 bg, past bg, bg_diff, past insulin, past carbs
        p_num = "p02"
        feat_arr = np.concatenate([
            [p_num], [self.bg_gap], [sin_t], [cos_t],
            bg_lags, bg_diff,
            insulin_hist, carbs_hist
        ]).reshape(1, -1)

        # 4) DataFrame으로 변환 (feature_names_in_ 순서 사용)
        input_cols = list(self.transformer.feature_names_in_)
        feat_df    = pd.DataFrame(feat_arr, columns=input_cols)
        for c in feat_df.columns:
            if c != "p_num":
                feat_df[c] = feat_df[c].astype(float)

        # 5) Transformer → 예측
        Xt = self.transformer.transform(feat_df)
        if isinstance(Xt, pd.DataFrame):
            Xt = Xt.astype(float)
        pred_bg = float(self.regressor.predict(Xt)[0])

        # 6) clipped_reward → 최종 보상
        new_reward = clipped_reward(pred_bg)
        new_reward = np.clip(new_reward, -1.0, 1.0)

        return obs, new_reward, terminated, truncated, info

def make_env():
    base = gym.make("simglucose-adol2-v0")
    hist = MultiHistoryWrapper(base, history_length=12)
    mon  = Monitor(hist)
    rew  = LGBMRewardWrapper(mon, model_path="lgbm_model.pkl")
    return rew

def main():
    os.makedirs("logs", exist_ok=True)

    # 1) DummyVecEnv으로 병렬 환경(n_envs=4) 생성
    vec_env = DummyVecEnv([make_env for _ in range(4)])

    # 2) PPO 에이전트 초기화 (TensorBoard 로그 지정)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./logs/"
    )

    # 3) 평가용 환경 및 콜백 설정
    eval_env = Monitor(make_env())
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True
    )

    # 4) 학습 (200k timesteps)
    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("ppo_simglucose_hist_tree_adol2")

    # 5) 학습된 모델 평가 (10 에피소드)
    mean_reward, std_reward = evaluate_policy(
        model, vec_env, n_eval_episodes=10, render=False
    )
    print(f"[Evaluation] Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # 6) 단일 환경으로 롤아웃 시연 (혈당 변화 시각화 예시)
    single = make_env()
    obs, info = single.reset(seed=123)
    bg_history = []
    time_steps = []
    max_steps = single.env.env.env.spec.max_episode_steps  # '!Monitor → MultiHistoryWrapper → 원 env' 순서로 감싸져 있으므로 .env.env.env

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = single.step(action)

        # info["bg"]를 통해 현재 혈당 추출
        bg = info["bg"]
        bg_history.append(bg)
        time_steps.append(t)

        if terminated or truncated:
            print(f"Demo finished at step {t+1}")
            break

    # 7) 혈당 변화 플롯 저장
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
