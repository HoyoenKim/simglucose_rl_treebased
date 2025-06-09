import os
import pickle
from collections import deque
from typing import Any, Deque, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Wrapper, ActionWrapper
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Beta
import torch as th
import torch.nn as nn

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
        return 1.0

    # below target
    if delta < -sigma:
        if delta <= -2 * sigma:
            return -5.0
        frac = (delta + 2 * sigma) / sigma
        return -5.0 + frac * 10.0

    # above target
    if delta > sigma:
        if delta >= 2 * sigma:
            return -2.0
        frac = (delta - sigma) / sigma
        return -2.0 - frac * 5.0

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

        current_bg = float(info["bg"])
        # 단위 변환 mmol/L -> mgd/L
        hist_bg = info["hist_bg"] / 18
        bg_lags = np.concatenate([[current_bg / 18], hist_bg])

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

        # 단위 변환 mmol/L <- mgd/L
        pred = float(self.regressor.predict(Xt)[0]) * 18
        reward = continuous_reward(pred, target=120.0, sigma=20.0)
        #print(pred, reward)
        return obs, reward, done, trunc, info


class SafetyFilter(ActionWrapper):
    """
    혈당이 cutoff 이하일 때 행동 스케일 조정
    """

    def __init__(
            self,
            env: gym.Env,
            scale: float = 0.1,
            cutoff: float = 80
    ):
        super().__init__(env)
        self.scale = scale
        self.cutoff = cutoff
        self._last_info: Dict[str, Any] = {}

    def action(self, action: Any) -> Any:
        bg = self._last_info.get("bg", None)
        if bg is not None and bg < self.cutoff:
            modified_action = np.asarray(action) * self.scale
            if isinstance(self.action_space, gym.spaces.Box):
                modified_action = np.clip(modified_action, self.action_space.low, self.action_space.high)
            return modified_action
        return action

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        safe_action = self.action(action)
        obs, reward, done, truncated, info = self.env.step(safe_action)
        self._last_info = info
        return obs, reward, done, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        return obs, info


# =============================================================================
# Custom Beta Distribution Policy for PPO
# =============================================================================
from torch.nn import functional as F
class CustomBetaPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        if not isinstance(self.action_space, gym.spaces.Box) or self.action_space.shape != (1,):
            raise ValueError("CustomBetaPolicy is designed for a single continuous action (insulin dose).")

        # Re-define the action_net to output 2 parameters (alpha, beta) for the Beta distribution.
        # The output dimension is 2 * action_dimension (which is 1 here)
        self.param_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 2)

    def _get_dist(self, latent_pi: th.Tensor) -> th.distributions.Distribution:
        raw = self.param_net(latent_pi)  # shape (batch, 2)
        alpha = F.softplus(raw[:, 0]) + 1.0  # shape (batch,)
        beta = F.softplus(raw[:, 1]) + 1.0  # shape (batch,)
        # (batch,) → (batch,1) 로 만들어 주고
        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return Beta(alpha, beta)

    def _get_action_dist_from_values(
            self,
            action_log_probs: th.Tensor,
            value: th.Tensor,
            latent_pi: th.Tensor,
            latent_vf: th.Tensor,
    ) -> Tuple[th.distributions.Distribution, th.Tensor, th.Tensor]:
        """
        Creates a Beta distribution for the policy's action.
        """
        raw_params = self.action_net(latent_pi)

        alpha = th.nn.functional.softplus(raw_params[:, 0]) + 1.0
        beta = th.nn.functional.softplus(raw_params[:, 1]) + 1.0

        # Ensure alpha and beta have the correct shape (batch_size, 1) if action_space.shape is (1,)
        # PyTorch Beta distribution can accept (batch_size,) for parameters if action_dim is 1,
        # but Stable-Baselines3's internal handling expects the sampled action to be (batch_size, action_dim).
        # We need to explicitly make alpha and beta (batch_size, 1) if action_dim is 1.
        alpha = alpha.unsqueeze(-1)  # Shape becomes (batch_size, 1)
        beta = beta.unsqueeze(-1)  # Shape becomes (batch_size, 1)

        dist = Beta(alpha, beta)

        return dist, value, None


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
    env = LGBMRewardWrapper(env)
    env = SafetyFilter(env)
    env = Monitor(env)
    return env


# =============================================================================
# Metrics & Evaluation
# =============================================================================
def compute_metrics_ppo(model: PPO, n_episodes: int = 20) -> Dict[str, float]:
    tir_count = tir_total = 0
    lbgi_list: list[float] = []
    hbgi_list: list[float] = []

    for ep in range(n_episodes):
        eval_env = make_env()
        obs, info = eval_env.reset(seed=42 + ep)

        lbgi_vals: list[float] = []
        hbgi_vals: list[float] = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)

            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()

            obs, _, term, trunc, info = eval_env.step(action)
            done = term or trunc

            bg = info["bg"]

            if isinstance(bg, np.ndarray):
                bg = float(bg.item())

            fbg = 1.509 * ((np.log(bg)) ** 1.084 - 5.381)
            rbg = 10 * (fbg ** 2)

            if fbg < 0:
                lbgi_vals.append(rbg)
            else:
                hbgi_vals.append(rbg)

            tir_total += 1
            if 70 <= bg <= 180:
                tir_count += 1
        eval_env.close()

        lbgi_list.append(np.mean(lbgi_vals) if lbgi_vals else 0.0)
        hbgi_list.append(np.mean(hbgi_vals) if hbgi_vals else 0.0)

    mean_lbgi = float(np.mean(lbgi_list)) if lbgi_list else 0.0
    mean_hbgi = float(np.mean(hbgi_list)) if hbgi_list else 0.0

    return {
        "Time-in-Range (%)": 100 * tir_count / tir_total if tir_total else 0.0,
        "LBGI (mean)": mean_lbgi,
        "HBGI (mean)": mean_hbgi,
    }


def evaluate_and_plot_ppo(
        model: PPO, n_eval: int = 20, ep_len: int = 288
) -> Dict[str, float]:
    eval_env = DummyVecEnv([make_env])
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval, deterministic=True, render=False
    )
    print(f"Eval {n_eval} eps: {mean_r:.2f} ± {std_r:.2f}")

    metrics = compute_metrics_ppo(model, n_episodes=n_eval)
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")

    env = make_env()
    obs, info = env.reset(seed=42)
    bg_traj: list[float] = []
    ins_traj: list[float] = []

    for _ in range(ep_len):
        action, _states = model.predict(obs, deterministic=True)
        ins_traj.append(float(np.array(action).ravel()[0]))

        bg_traj.append(info["bg"])

        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    env.close()

    plt.figure()
    plt.plot(bg_traj, marker="o", linestyle='-')
    plt.title("BG Trajectory")
    plt.xlabel("Timestep (5 min)")
    plt.ylabel("BG (mg/dL)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_result_bg.png")
    plt.close()

    plt.figure()
    plt.plot(ins_traj, marker="o", linestyle='-')
    plt.title("Insulin Trajectory")
    plt.xlabel("Timestep (5 min)")
    plt.ylabel("Insulin (IU)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_result_ins.png")
    plt.close()

    return metrics


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    register_env()
    best_model_path = os.path.join(BEST_MODEL_DIR, "best_model")
    if os.path.exists(f"ppo_simglucose_hist_tree_adol2.zip"):
        print(f"Loading model from ppo_simglucose_hist_tree_adol2.zip")

        eval_vec_env_for_loading = DummyVecEnv([make_env])
        eval_vec_env_for_loading = VecNormalize(eval_vec_env_for_loading, norm_obs=True, norm_reward=False,
                                                clip_obs=10.)
        eval_vec_env_for_loading.seed(42)
        loaded_model = PPO.load(best_model_path, env=eval_vec_env_for_loading)
        evaluate_and_plot_ppo(loaded_model)
    else:
        print("No ppo_simglucose_hist_tree_adol2.zip found.")

if __name__ == "__main__":
    main()