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

def continuous_reward(pred_bg: float,
                      target: float = 125.0,
                      sigma: float = 25.0) -> float:
    """
    Gaussian-shaped reward centered at `target`.
    `sigma`가 작을수록 보상 곡선이 날카로워집니다.
    반환값은 (0, 1] 범위입니다.
    """
    return float(np.exp(-0.5 * ((pred_bg - target) / sigma) ** 2))

# 1) simglucose 환경 등록
register(
    id="simglucose-adol2-v0",
    entry_point="simglucose.envs:T1DSimGymnaisumEnv",
    max_episode_steps=200,  # 에피소드 최대 길이 (원하는 값으로 조정)
    kwargs={
        "patient_name": "adolescent#002",
        # "custom_scenario": CustomScenario(start_time=datetime(2025,5,16), scenario=[(60,20)]),
        # "reward_fun": custom_reward,  # 필요 시 사용자 정의 보상 함수
    },
)

# 1) MultiHistoryWrapper: bg, meal, action 히스토리(12스텝) 생성
class MultiHistoryWrapper(Wrapper):
    def __init__(self, env, history_length: int = 12):
        super().__init__(env)
        self.history_length = history_length
        self.bg_buf      = deque(maxlen=history_length)
        self.meal_buf    = deque(maxlen=history_length)
        self.insulin_buf = deque(maxlen=history_length)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 초기값으로 버퍼 채우기
        init_bg   = info["bg"]
        init_meal = info["meal"]
        init_ins  = 0.0  # reset 직후엔 action 없음 → 0
        for _ in range(self.history_length):
            self.bg_buf.append(init_bg)
            self.meal_buf.append(init_meal)
            self.insulin_buf.append(init_ins)
        # info 에 히스토리 추가
        info["hist_bg"]      = np.array(self.bg_buf)       # shape=(12,)
        info["hist_meal"]    = np.array(self.meal_buf)     # shape=(12,)
        info["hist_insulin"] = np.array(self.insulin_buf)  # shape=(12,)
        return obs, info

    def step(self, action):
        # action 값도 히스토리에 기록
        obs, reward, terminated, truncated, info = self.env.step(action)
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

# 2) LightGBM 기반 보상 래퍼
class LGBMRewardWrapper(Wrapper):
    def __init__(self, env, model_path: str):
        super().__init__(env)
        # 1) 파이프라인 전체(Transformer + Regressor) 로드
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        # 2) Transformer와 Regressor 분리
        self.transformer = self.pipeline.named_steps["transform"]
        self.regressor   = self.pipeline.named_steps["regressor"]
        self.bg_gap = 5

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # —————————————— 1) 원본 X 배열 생성 ——————————————
        p_num = "p02"
        t = info["time"]
        minutes = t.hour * 60 + t.minute
        sin_t = np.sin(2 * np.pi * minutes / 1440)
        cos_t = np.cos(2 * np.pi * minutes / 1440)

        hist_bg     = info["hist_bg"]
        current_bg  = float(info["bg"])
        bg_lags     = np.concatenate([[current_bg], hist_bg])  # (13,)
        bg_diff     = np.diff(bg_lags)                         # (12,)
        insulin_hist= info["hist_insulin"]                     # (12,)
        carbs_hist  = info["hist_meal"]                        # (12,)

        feat_arr = np.concatenate([
            [p_num], [self.bg_gap], [sin_t], [cos_t],
            bg_lags, bg_diff,
            insulin_hist, carbs_hist
        ]).reshape(1, -1)

        # —————————————— 2) 원래 기대 피처명으로 DataFrame 생성 ——————————————
        input_cols = list(self.transformer.feature_names_in_)
        feat_df    = pd.DataFrame(feat_arr, columns=input_cols)

        # 모든 행·열, 인덱스 없이 출력
        # pd.set_option("display.max_rows", None)
        # pd.set_option("display.max_columns", None)
        # print(feat_df.to_string(index=False))

        # —————————————— 3) 숫자형 칼럼들 float 으로 캐스팅 ——————————————
        #    (p_num 은 object, 나머지는 모두 float)
        for c in feat_df.columns:
            if c != "p_num":
                feat_df[c] = feat_df[c].astype(float)

        # —————————————— 4) 인코더 → 예측 ——————————————
        Xt = self.transformer.transform(feat_df)
        # transformer 출력도 object dtype 이 남아 있을 수 있으므로,
        # Xt 가 DataFrame 이면 전체를 float 으로 바꿔줍니다
        if isinstance(Xt, pd.DataFrame):
            Xt = Xt.astype(float)

        pred_bg = float(self.regressor.predict(Xt)[0])
        new_reward = continuous_reward(pred_bg,
                                       target=125.0,  # 목표 혈당
                                       sigma=25.0)  # 곡선 폭
        return obs, new_reward, terminated, truncated, info

# 3) 서브환경 팩토리
def make_env():
    base = gym.make("simglucose-adol2-v0")
    hist = MultiHistoryWrapper(base, history_length=12)
    mon = Monitor(hist)
    rew  = LGBMRewardWrapper(hist, model_path="lgbm_model.pkl")
    return rew

def main():
    os.makedirs("logs", exist_ok=True)

    # 3) DummyVecEnv으로 병렬 환경(n_envs=4) 생성
    vec_env = DummyVecEnv([make_env for _ in range(4)])

    # 4) PPO 에이전트 초기화 (TensorBoard 생략)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./logs/",
    )

    # 5) 학습 (200k timesteps)
    eval_env = Monitor(make_env())
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/best_model",
                                 log_path="./logs/eval", eval_freq=10_000,
                                 n_eval_episodes=5, deterministic=True)

    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("ppo_simglucose_hist_tree_adol2")

    # 6) 평가 (10 에피소드)
    mean_reward, std_reward = evaluate_policy(
        model, vec_env, n_eval_episodes=10, render=False
    )
    print(f"[Evaluation] Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # 7) 시연 (단일 env + Wrapper 체인)
    single = make_env()
    obs, info = single.reset(seed=123)
    for t in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = single.step(action)
        single.render()
        if terminated or truncated:
            print(f"Demo finished at step {t+1}")
            break
    single.close()

if __name__ == "__main__":
    main()