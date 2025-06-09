import gymnasium as gym
from gymnasium.envs.registration import register

import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
policy_path = "ppo_simglucose_hist_tree_adol7/policy.pth"
optimizer_path = "ppo_simglucose_hist_tree_adol7/policy.optimizer.pth"

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
bg_hist, insulin_hist, t_hist = [], [], []

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
    bg_hist.append(obs[0])        # 현재 혈당
    insulin_hist.append(action[0])  # 인슐린 주입량 (Basal U/5min)
    t_hist.append(t)
    if done or truncated:
        break

env.close()

# ─────────────────────────────────────────────────────────────────────────
# 5) BG + Insulin 애니메이션 생성 & GIF 저장
# ─────────────────────────────────────────────────────────────────────────
fig, (ax_bg, ax_ins) = plt.subplots(2, 1, sharex=True,
                                    figsize=(8, 6), dpi=100)

line_bg,  = ax_bg.plot([], [], lw=2, label="BG (mg/dL)")
line_ins, = ax_ins.plot([], [], lw=2, label="Insulin (U)")

ax_bg.set_ylabel("BG (mg/dL)")
ax_bg.set_ylim(min(bg_hist) - 20, max(bg_hist) + 20)
ax_bg.grid(True)
ax_bg.legend(loc="upper right")

ax_ins.set_xlabel("Time step")
ax_ins.set_ylabel("Insulin (U)")
ax_ins.set_ylim(min(insulin_hist) - 0.01, max(insulin_hist) + 0.01)
ax_ins.grid(True)
ax_ins.legend(loc="upper right")

def update(frame):
    line_bg.set_data(t_hist[:frame+1], bg_hist[:frame+1])
    line_ins.set_data(t_hist[:frame+1], insulin_hist[:frame+1])
    ax_bg.set_xlim(0, t_hist[frame])
    return line_bg, line_ins

ani = animation.FuncAnimation(fig, update,
                              frames=len(t_hist),
                              interval=150, blit=True)

ani.save("bg_insulin.gif", writer="pillow")  # pillow가 설치돼 있어야 함
plt.close(fig)

print("Saved BG + Insulin animation → bg_insulin.gif")


# 4) 결과 플롯
plt.figure(figsize=(10, 5))
plt.plot(time_steps, bg_history, marker='o')
plt.xlabel("Time step")
plt.ylabel("Blood Glucose (mg/dL)")
plt.title("Blood Glucose Control under Learned PPO Policy")
plt.grid(True)
plt.tight_layout()
plt.savefig("bg_control_plot_7.png")
plt.close()

print("Saved blood glucose control plot to bg_control_plot.png")