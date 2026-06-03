# Blood Glucose Prediction & Control in Type 1 Diabetes

An end-to-end pipeline for predicting future blood glucose (BG) levels with LightGBM and controlling BG via reinforcement learning (SAC & PPO) using the Sim-glucose OpenAI Gym environment.

---

## 🧠 Model Overview

### LightGBM Regressor
![LGBM_model2](./Result/Figure/LGBM_model2.png)
*Fig1. LightGBM model architecture*

* Learns to predict future BG using tabular features from the last hour of CGM, insulin, meals, and wearable sensor data (heart rate, steps, calories).
* **Algorithm**: Gradient boosting with decision trees (`LGBMRegressor`).
* **Objective**: Minimize RMSE on held-out time-series folds (expanding-window CV).

![LGBM_model1](./Result/Figure/LGBM_model1.png)
*Fig2. LightGBM model inference BG+1:00*

* **Inference loop**: At each timestep, the RL agent queries the predictor to estimate BG\_{t+1} given current state.

### SAC & PPO Agent

![RL_model](./Result/Figure/RL_model.png)
*Fig3. SAC & PPO RL model architecture*

#### SAC Baseline Agent

* Off-policy actor-critic algorithm that maximizes expected reward minus an entropy regularization term.
* **Reward**: Risk-delta (difference in LBGI/HBGI indices) with heavy penalties for hypo (<70 mg/dL) and hyper (>180 mg/dL) events.
* **Filter**: Rule-based safety filter blocks excessive insulin when BG drops below threshold.

#### PPO + LightGBM Agent

* On-policy proximal policy optimization wrapped around a Beta action distribution for continuous insulin dosing.
* **Integrated Predictor**: the LightGBM 1-hour BG forecast feeds into both the PPO **reward** and the policy **observation**. (In the original code the forecast never reached the observation due to a `pred_bg30`/`pred_bg60` key mismatch — fixed on branch `claude/bugfix-smoke`; see `CLAUDE.md` §7–§8.)
* **Reward**: Custom continuous (Gaussian-shaped) function penalizing deviations from target **140 mg/dL** (σ = 30 mg/dL), with asymmetric tail scaling beyond ±2σ that penalizes hypoglycemia far more strongly than hyperglycemia. (The README previously said 125 mg/dL; the code uses 140 — now pinned to explicit constants.)
* **Policy**: MLP policy outputs insulin dose in \[0,1] normalized units.

---

## 🔗 References

* Kaggle BRIST1D 1st Solution: [https://www.kaggle.com/competitions/brist1d/discussion/555236](https://www.kaggle.com/competitions/brist1d/discussion/555236)
* BRIST1D 1st Solution Repo: [https://github.com/scuya2050/brist1d\_blood\_glucose\_prediction\_competition](https://github.com/scuya2050/brist1d_blood_glucose_prediction_competition)
* Sim-glucose Gym Env: [https://github.com/jxx123/simglucose](https://github.com/jxx123/simglucose)
* PPO controller, “A safe-enhanced fully closed-loop artificial pancreas controller based on deep reinforcement learning”, 2025:
  [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0317662](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0317662)

---

## 📂 Repository Structure

```
.
├── data/                              # Kaggle dataset
├── prepare_data.py                    # Kaggle dataset preprocessing script
├── train.py                           # LightGBM training script
├── lgbm_model.pkl                     # trained LightGBM model
├── baseline.ipynb                     # SAC & PPO baseline agent notebook
├── simglucose_ppo_lgbm.py             # PPO + LightGBM training & evaluation script
├── ppo_simglucose_hist_tree_adol2.zip # PPO Model
├── logs/                              # PPO Model train logs
├── Result/                            # archive results
├── doc/                               # midterm, final PPT
└── requirements.txt                   # Python dependencies
```

---

## ⚙️ Environment Setup

1. Clone this repository and enter its root directory.

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the following files in place:

   * `prepare_data.py`, `train.py`
   * `simglucose_ppo_lgbm.py`, `baseline.ipynb`
   * `lgbm_model.pkl` (trained LightGBM regressor)
   * `ppo_simglucose_hist_tree_adol2.zip` (trained PPO agent)

4. Python 3.12+ is required.

---

## 🗄️ Data Preparation & LightGBM Model

### 0. Download Kaggle Dataset
* Download 'BrisT1D Blood Glucose Prediction Competition' dataset.
  * https://www.kaggle.com/competitions/brist1d/data

* Place them at ./data/raw.

### 1. Preprocess Kaggle BRIST1D Dataset

```
python prepare_data.py
```

* **Data sources**

  * CGM readings (5-minute intervals)
  * Self-reported carbs & activity
  * Insulin pump doses
  * Wearable sensor data (heart rate, steps, calories)

* **Preprocessing steps**

  * Sliding window expansion (12 lags × 5 min = 1 h history)
  * Time-of-day trigonometric encoding (sin/cos)
  * Forward/backward interpolation for missing CGM, insulin, HR, calories
  * Zero-fill carbs & activity flags

### 2. Train & Serialize LightGBM Regressor

```
python train.py
```

* **Model**: `LGBMRegressor`
* **Features**:

  * BG, insulin, carbs, HR, steps, calories at t−{0…60 min}
  * Patient statistics (mean, std, skewness, kurtosis)
* **Validation**: TabularExpandingWindowCV (time-aware CV)
* **Metric**: RMSE

### 3. Optional: Place the resulting model file
* Already included in the project.
```
cp ./model/lgbm_gap_1_prior_12_addition_0_model_standard.pkl ./lgbm_model.pkl
```

---

## 🤖 Reinforcement Learning Control

### 3. Train RL Agent

* **Agents**:

  1. **SAC** (baseline, in `baseline.ipynb`)

     * Risk-delta reward
     * Rule-based action filter

  2. **PPO + LightGBM** (in `simglucose_ppo_lgbm.py`)

     * LightGBM predictor in the loop
     * Custom continuous reward (penalize hypo/hyper)

```
python simglucose_ppo_lgbm.py train
```

*Baseline SAC agent training and analysis available in `baseline.ipynb`.*

#### 3.1. Check TensorBoard

```
tensorboard --logdir ./logs
```

* Browse at [http://localhost:6006/#timeseries](http://localhost:6006/#timeseries)

### 4. Evaluate & Visualize

```
python simglucose_ppo_lgbm.py eval --episodes 20 
```

#### 4.1 Key Metrics — bug-fixed pipeline (patient `adolescent#002`, 20 eval seeds)

All rows are **re-validated** on the bug-fixed pipeline with the corrected evaluation
(observations normalized to training stats; 20 distinct seeds; `±` = spread across seeds).
PPO configs use the full 2.3M-step budget; SAC uses 100k (its notebook budget); filter-only
needs no training. See `CLAUDE.md` §7–§8 for the bugs and the verification log.

| Controller | TIR (%) | LBGI | HBGI | Time-below-54 (%) |
| --- | ---: | ---: | ---: | ---: |
| Filter-only (no RL, zero policy) | 23.49 ± 7.28 | 0.00 | 39.51 | 0.00 |
| SAC + risk-delta (no filter) | 56.84 ± 13.43 | 45.53 | 2.37 | **38.01** |
| SAC + risk-delta + filter | 26.47 ± 11.34 | 0.00 | 39.23 | 0.00 |
| PPO + LightGBM + Reward (no filter) | 74.92 ± 12.92 | 14.14 | 6.04 | 2.15 |
| **PPO + LightGBM + Reward (+ filter)** | **77.80 ± 11.91** | 17.43 | 4.72 | 3.90 |

> **Read LBGI / time-below-54, not TIR alone.**
>
> - **The learned policy — not the safety filter — drives the result.** PPO+LGBM+Reward reaches
>   ~75% TIR *without* any filter; adding the filter only nudges TIR 75→78% while *raising*
>   hypoglycemia (LBGI 14→17, time<54 2.15→3.90%). The original "the filter does the work" story
>   does **not** hold for the fixed pipeline — here the filter is even mildly counter-productive
>   for safety.
> - **Filter-only ≈ 23%** (no dosing → hyperglycemia): the filter alone is just a safety overlay.
> - **SAC is unstable / unsafe here.** Across single training seeds it swings between under-dosing
>   (26% TIR, hyperglycemia) and over-dosing (57% TIR but **38% of time < 54 mg/dL** — severe
>   hypoglycemia). The original single-seed SAC numbers (23% / 61%) are not reproducible.
> - **No config is clinically safe yet:** even the best (77.8% TIR) has LBGI ≈ 17 and ~4% time
>   < 54 mg/dL (consensus target < 1%). **High TIR ≠ safe.**
>
> *(The original "PPO" (no-LGBM) and "PPO + LightGBM" (no custom reward) rows came from the buggy
> pipeline and are superseded; re-validating them would need reward-variant flags plus another
> full LGBM training run.)*

#### 4.2 BG & Insulin Trajectories

- The figures below are from the **bug-fixed full model** (PPO + LightGBM + Reward, 2.3M steps, patient `adolescent#002`, single representative seed). Note the BG trace still dips into hypoglycemia — consistent with the LBGI ≈ 17 / 3.9 % time-below-54 reported above.
- Since the simulator has randomness, the trajectory changes each run.

![BG Trajectory](./Result/Figure/PPO_LGBM_reward_result_bg.png)

*Fig 4. Future BG vs. time under PPO + LGBM + custom reward*

![Insulin Trajectory](./Result/Figure/PPO_LGBM_reward_result_ins.png)

*Fig 5. Administered insulin over time*

---

## 🚀 Future Work

* Apply **Transformer-based predictor**: pre-trained on simulator data
* Apply **Advanced PPO controller**: leveraging Beta policy & advanced stabilization tricks

---