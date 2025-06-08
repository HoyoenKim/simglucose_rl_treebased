# -*- coding: utf-8 -*-
"""
Baseline: SAC on simglucose (Risk-Δ 보상)
========================================

• 환경:  simglucose T1DSimGymnaisumEnv (기본 Risk-Δ 보상 사용)
• 알고리즘: SAC (off-policy, 샘플 효율↑으로 PPO보다 빠른 학습)
• 목표: PPO 대비 수렴 속도 및 최종 성능 비교
"""

import os
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# --------------------------------------------------------------------- #
# 1. simglucose 환경 등록 (risk Δ 보상은 내부 기본값 그대로 사용)
# --------------------------------------------------------------------- #
register(
    id="simglucose-adol2-v0",
    entry_point="simglucose.envs:T1DSimGymnaisumEnv",  # 실제 클래스명 확인
    max_episode_steps=288,                 # 1일(5-min step × 288)
    kwargs={
        "patient_name": "adolescent#002",  # baseline 설정: 동일 환자 프로파일
        # reward_fun을 지정하지 않으면 → 기본 risk Δ 보상 사용
    },
)

# --------------------------------------------------------------------- #
# 2. 환경 팩토리 (Monitor만 래핑)
# --------------------------------------------------------------------- #
def make_env():
    env = gym.make("simglucose-adol2-v0")
    env = Monitor(env)          # episode reward, length 로깅
    return env

# --------------------------------------------------------------------- #
# 3. 메인 함수: SAC 학습·평가
# --------------------------------------------------------------------- #
def main():
    log_dir = Path("logs_sac_baseline")
    log_dir.mkdir(exist_ok=True)

    # ── 병렬 환경 4개 생성 ────────────────────────────────────────────────
    vec_env = DummyVecEnv([make_env for _ in range(4)])

    # ── SAC 에이전트 초기화 (TensorBoard 로그 경로 지정) ────────────────
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",         # 자동 온도 계수 조절
        verbose=1,
        tensorboard_log=str(log_dir),  # TensorBoard 로그 저장
    )

    # ── 평가 콜백: 10k 스텝마다 5 에피소드 평가 ───────────────────────────
    eval_env = Monitor(make_env())
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # ── 학습 (200k timesteps) ────────────────────────────────────────────
    model.learn(total_timesteps=20_000, callback=eval_cb)
    model.save("sac_risk_baseline")

    return model

# --------------------------------------------------------------------- #
# 4. 평가 및 데모
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    sac_model = main()

    # ── 최종 평가 (10 에피소드) ─────────────────────────────────────────
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    mean_reward, std_reward = evaluate_policy(
        sac_model, vec_env, n_eval_episodes=10, render=False
    )
    print(f"[Baseline SAC Evaluation] Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # ── 데모 + GIF 저장(Optional) ─────────────────────────────────────────
    try:
        import imageio

        def demo_and_save_gif(model, save_path="sac_baseline_demo.gif",
                              max_steps=288, fps=5):
            env = make_env()
            obs, _ = env.reset(seed=42)
            frames = []
            for t in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                frame = env.render(mode="rgb_array")
                frames.append(frame)
                if term or trunc:
                    break
            env.close()
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"[GIF saved] {save_path}")

        demo_and_save_gif(sac_model, save_path="sac_baseline_demo.gif", fps=5)
    except ImportError:
        print("imageio가 설치되어 있지 않습니다. GIF 저장을 건너뜁니다.")
