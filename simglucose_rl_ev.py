import gymnasium as gym
from gymnasium.envs.registration import register

import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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

# 1) 환경 생성
env = gym.make("simglucose-adol2-v0")

# 2) 동일한 하이퍼파라미터로 PPO 모델 초기화 (더미로 선언)
model = PPO("MlpPolicy", env, verbose=0)

# —————————— 변경된 부분: PPO.load() 사용 ——————————
model = PPO.load("ppo_simglucose_hist_tree_adol2", env=env, map_location="cpu")
# ——————————————————————————————————————————————

# 4) 평가
n_eval_episodes = 20
rewards, lengths = evaluate_policy(
    model,
    env,
    n_eval_episodes=n_eval_episodes,
    return_episode_rewards=True
)

# 5) 결과 플롯 저장
plt.figure()
plt.plot(rewards, marker="o")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Evaluation on simglucose-adol2-v0")
plt.grid(True)
plt.tight_layout()
out_path = "ppo_eval_rewards.png"
plt.savefig(out_path)
plt.close()

print(f"Evaluation plot saved to {out_path}")

# 6) 에피소드 초기화 (롤아웃)
obs, info = env.reset()
bg_history = []
time_steps = []

# 최대 스텝 수만큼 시뮬레이션
max_steps = env.spec.max_episode_steps

for t in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # 더 안전하게 blood glucose를 가져오려면 info["bg"] 사용
    bg = info["bg"]  # 또는 obs[0]
    bg_history.append(bg)
    time_steps.append(t)

    if done or truncated:
        break

# 7) 혈당 변화 플롯
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
