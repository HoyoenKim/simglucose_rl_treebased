import os
import pickle
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

# 1) HistoryWrapper: info['patient_state'] 시계열을 info['state_hist']로 축적
class HistoryWrapper(Wrapper):
    def __init__(self, env, history_length: int = 12):
        super().__init__(env)
        self.history_length = history_length
        self.buffer = deque(maxlen=history_length)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 초기 상태로 history 채우기
        init = info["patient_state"].copy()
        self.buffer.clear()
        for _ in range(self.history_length):
            self.buffer.append(init)
        info["state_hist"] = np.stack(self.buffer, axis=0)  # (12, n_feats)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 최신 상태 추가
        self.buffer.append(info["patient_state"].copy())
        info["state_hist"] = np.stack(self.buffer, axis=0)
        return obs, reward, terminated, truncated, info

# 2) LightGBM 기반 보상 래퍼
class LGBMRewardWrapper(RewardWrapper):
    def __init__(self, env, model_path: str):
        super().__init__(env)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        # pickle 또는 joblib 로드
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # print(type(info), info.keys() if isinstance(info, dict) else info)
        # <class 'dict'> dict_keys(['sample_time', 'patient_name', 'meal', 'patient_state', 'time', 'bg', 'lbgi', 'hbgi', 'risk', 'episode'])

        # 1) patient_state 꺼내기
        ps = info["patient_state"]

        # 2) 주요 피처 추출
        # basal_hist = ps["basal"][-12:]
        # bolus_hist = ps["bolus"][-12:]
        # meal_hist  = ps["meal"][-12:]

        # 3) 피처 벡터 합치기 (여기서는 CGM만 사용)
        X = ps.reshape(1, -1)

        # 4) LightGBM 모델 예측 → 보상으로 사용
        new_reward = float(self.model.predict(X)[0])

        # 5) 새로운 reward와 함께 리턴
        return obs, new_reward, terminated, truncated, info

def make_env():
    """각 서브환경 초기화: base → History → RewardWrapper 체인"""
    base = gym.make("simglucose-adol2-v0")
    hist = HistoryWrapper(base, history_length=12)
    rew  = LGBMRewardWrapper(hist, model_path="lgbm_gap_1_prior_12_addition_0_model_standard.pkl")
    return rew

def main():
    # 3) DummyVecEnv으로 병렬 환경(n_envs=4) 생성
    vec_env = DummyVecEnv([make_env for _ in range(4)])

    # 4) PPO 에이전트 초기화 (TensorBoard 생략)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=None,
    )

    # 5) 학습 (200k timesteps)
    model.learn(total_timesteps=200_000)
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