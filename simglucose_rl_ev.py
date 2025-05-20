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

# 2) 동일한 하이퍼파라미터로 PPO 모델 초기화 (정상적으로 동작하는 더미 정책)
model = PPO("MlpPolicy", env, verbose=0)

# 3) 정책 네트워크 & 옵티마이저 파라미터 로드
device = torch.device("cpu")
policy_path = "ppo_simglucose_hist_tree_adol2/policy.pth"
optimizer_path = "ppo_simglucose_hist_tree_adol2/policy.optimizer.pth"

# (a) policy network state_dict 로드
policy_state = torch.load(policy_path, map_location=device)
model.policy.load_state_dict(policy_state)

# (b) optimizer state_dict 로드
opt_state = torch.load(optimizer_path, map_location=device)
model.policy.optimizer.load_state_dict(opt_state)

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

# 3) 에피소드 초기화
obs, info = env.reset()
bg_history = []
time_steps = []

# 최대 스텝 수만큼 시뮬레이션
max_steps = env.spec.max_episode_steps

for t in range(max_steps):
    # 정책에 따라 행동 결정 (deterministic=True 로 결정론적 행동)
    action, _ = model.predict(obs, deterministic=True)

    # 환경에 한 스텝 진행
    obs, reward, done, truncated, info = env.step(action)

    # obs 배열의 첫 번째 원소가 현재 혈당(glucose) 값이라고 가정
    bg = obs[0]
    bg_history.append(bg)
    time_steps.append(t)

    if done or truncated:
        break

# 4) 결과 플롯
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