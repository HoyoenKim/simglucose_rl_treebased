# CLAUDE.md

Guidance for Claude Code when working in this repository. It documents **what has been
built**, **how to run it**, and a **critical assessment with concrete improvement areas**.

---

## 1. What this project is

A coursework / research project (see `doc/` for the midterm + final slides) on
**blood-glucose (BG) prediction and control for Type-1 Diabetes**. It chains two stages:

1. **LightGBM BG predictor** — a tabular gradient-boosting regressor that forecasts BG
   **1 hour ahead** (`bg+1:00`) from the last hour of CGM / insulin / carb (and optionally
   wearable) history. Adapted from the Kaggle **BrisT1D** 1st-place solution
   (`scuya2050/brist1d_blood_glucose_prediction_competition`).
2. **RL controller** — agents that dose insulin inside the **simglucose** T1D simulator
   (OpenAI Gym / Gymnasium). Two variants:
   - **SAC baseline** (`baseline.ipynb`): risk-delta reward + rule-based safety filter.
   - **PPO + LightGBM** (`simglucose_ppo_lgbm.py`): a custom Beta-policy PPO whose reward
     is shaped by the LightGBM BG forecast plus a Gaussian target reward.

The headline claim is that **PPO + LightGBM + custom reward** reaches the best
Time-in-Range (≈68%). **Read §7 before trusting that number.**

---

## 2. Repository layout

```
prepare_data.py              # Stage-1 data prep (BrisT1D Kaggle CSV -> processed pickles)
train.py                     # Stage-1 LightGBM training (Optuna tuning + expanding-window CV)
brist1d/                     # Feature engineering + CV + encoders (from BrisT1D solution)
  ├─ tabular_transformers.py # lag/diff/interp feature builders, data_expander
  ├─ pipeline_transformers.py# per-patient target encoders (mean/std/skew/kurt)
  ├─ cross_validator.py      # TabularExpandingWindow{Splitter,CV}
  ├─ params.py               # Stage-1 hyperparameters / run flags
  └─ utils.py                # paths from settings.json, timer, logger
lgbm_model.pkl               # trained Stage-1 pipeline (transform + regressor)  [13.8 MB]
simglucose_ppo_lgbm.py       # Stage-2 PPO+LGBM: wrappers, Beta policy, train/eval CLI
baseline.ipynb               # Stage-2 SAC baseline + experiments (15 cells)
ppo_simglucose_hist_tree_adol2.zip  # trained PPO checkpoint
logs/                        # TensorBoard event files (PPO_134, PPO_138)
Result/                      # figures, Result_Summary.txt, duplicate model copy
doc/                         # midterm + final PDFs
settings.json                # data/model directory paths (Stage-1)
requirements.txt             # pinned deps (NOTE: UTF-16 encoded, full pip freeze)
```

Stage 1 (LightGBM) and Stage 2 (RL) are **loosely coupled**: the only artifact passed
between them is `lgbm_model.pkl`, loaded by `LGBMRewardWrapper`.

---

## 3. Commands

All Stage-1 commands must be run **from the repo root** (see §6 — `utils.py` opens
`./settings.json` with a relative path at import time).

```bash
# Stage 1 — data prep + LightGBM (needs Kaggle BrisT1D train.csv/test.csv in data/raw/)
python prepare_data.py            # reads brist1d/params.py for GAP/N_PRIOR/...
python train.py                   # Optuna tuning + fit; writes model/<...>.pkl
# then manually: cp model/lgbm_gap_1_prior_12_addition_0_model_standard.pkl lgbm_model.pkl

# Stage 2 — PPO + LightGBM controller
python simglucose_ppo_lgbm.py train          # train/continue PPO (writes checkpoint + stats)
python simglucose_ppo_lgbm.py eval --episodes 20
tensorboard --logdir ./logs                  # http://localhost:6006

# Stage 2 SAC baseline: open baseline.ipynb
```

There is **no test suite, linter, or CI**. Python 3.12+; key pins: `stable_baselines3==2.6.0`,
`gymnasium==0.29.1` (and a stray `gym==0.9.4`), `simglucose==0.2.11`, `lightgbm==4.6.0`,
`torch==2.7.0`.

---

## 4. Stage-2 architecture (most active code)

`env_factory()` builds this wrapper stack (outer → inner):

```
Monitor → FeatureObsWrapper → SafetyFilter → LGBMRewardWrapper → MultiHistoryWrapper → T1DSimGymnasiumEnv
```

- **MultiHistoryWrapper** — keeps 1 h (12×5 min) deques of BG, meal, insulin; exposes them in `info`.
- **LGBMRewardWrapper** — builds a feature row, runs `lgbm_model.pkl`, and replaces the
  env reward with `continuous_reward(pred_bg) + λ_risk·Δrisk − λ_ins·action`, clipped to ±5.
  Falls back to a 2 h linear extrapolation when the LGBM jump looks implausible (|Δ|≥50).
- **SafetyFilter** (`ActionWrapper`) — rule-based override: zero insulin when BG<80 or BG
  falling; force a min bolus when BG>250.
- **FeatureObsWrapper** — flattens obs + histories + sin/cos time + predicted BG into a vector.
- **CustomBetaPolicy** — PPO head emitting (α,β) of a Beta distribution for `Box(1,)` dosing.

Training: 4 parallel envs, `VecNormalize`, `n_steps=288` (one simulated day), 2.3 M steps,
single patient `adolescent#002`. Eval reports mean reward + custom TIR / LBGI / HBGI.

---

## 5. Key conventions

- **Units:** simglucose BG is mg/dL; the BrisT1D LightGBM works in mmol/L → conversion factor
  `18.0182` is applied in `LGBMRewardWrapper.step`.
- **Risk index:** Kovatchev BG-risk function `f(bg)=1.509·(ln(bg)^1.084 − 5.381)`, `risk=10·f²`,
  split into LBGI (f<0) / HBGI (f>0). Reused (copy-pasted) in both the wrapper and `evaluate_agent`.
- **Stage-1 file naming:** artifacts are keyed by `gap_{GAP}_prior_{N_PRIOR}_addition_{ADDITION}`
  and a `SUFFIX` (`standard`). Change runs via `brist1d/params.py`, not CLI flags.
- **Paths:** Stage-1 dirs come from `settings.json`; most (`model/`, `work/`, `hyperparameter_tuning/`)
  are git-ignored, so a fresh clone only has the pre-built `lgbm_model.pkl`.

---

## 6. Gotchas (things that will bite you)

- `brist1d/utils.py` does `json.load(open("./settings.json"))` **at import time** → importing
  anything from `brist1d/` outside the repo root raises `FileNotFoundError`.
- `.gitignore` has `__pycahce__/` (typo) so `__pycache__/` is **not** ignored; and `*.txt`
  is ignored even though `requirements.txt`/`Result_Summary.txt` are tracked.
- `requirements.txt` is **UTF-16-encoded** and is a full `pip freeze` (100+ transitive pins,
  including Jupyter and both `gym` and `gymnasium`). `pip install -r` may misbehave.
- `vec_normalize_stats.pkl` is referenced by `train`/`eval` and whitelisted in `.gitignore`
  but is **not committed** — see §7 #2.

---

## 7. Critical assessment & improvement areas

Ordered by severity. File:line references are to `simglucose_ppo_lgbm.py` unless noted.

### A. Correctness bugs that affect the reported results

1. **Dead predicted-BG feature (key mismatch).** `LGBMRewardWrapper` writes the forecast as
   `pred_bg60` (lines 147, 191) but `FeatureObsWrapper` reads `info.get("pred_bg30", obs[-1])`
   (line 250). `pred_bg30` is never set, so the policy's observation **always uses the fallback
   `obs[-1]`** — the LightGBM prediction reaches the *reward* but never the *observation*. The
   whole "predictor-in-the-loop for the policy" premise is silently broken. → Fix the key.

2. **Evaluation runs without the training-time normalization.** The model is trained under
   `VecNormalize(norm_obs=True)`, but (a) `evaluate_agent`'s custom TIR/LBGI/HBGI loop and the
   trajectory plots use a **raw** `env_factory()` (lines 453, 474) instead of the normalized
   vec-env, and (b) `vec_normalize_stats.pkl` is **not committed**, so even `evaluate_policy`
   loads an identity normalizer. The policy therefore sees out-of-distribution observation
   scales in exactly the loop that produces the headline numbers. → Persist + load the stats,
   and evaluate through the same `VecNormalize`.

3. **Reported metrics are effectively n=1.** `Result_Summary.txt` shows `± 0.00` std across
   "20 episodes", and `evaluate_agent` resets every episode with the **same** `seed=SEED`
   (line 454). Deterministic policy + identical seed ⇒ identical episodes. The "20-episode"
   evaluation has no real variance. → Use distinct seeds / un-seeded resets and report CIs.

### B. Methodology / scientific validity

4. **Real-data predictor used on synthetic data (domain shift).** The LightGBM is trained on
   real human BrisT1D data but queried on simglucose's synthetic output, including a
   **hardcoded patient id `"p02"`** (line 162) that arbitrarily maps `adolescent#002` to a
   Kaggle patient's per-patient encoder statistics. Insulin history fed to the model is the
   agent's normalized `[0,1]` action, not clinical units. The predictor is run far
   out-of-distribution; its forecasts are of unknown validity here.

5. **Horizon labeling is inconsistent.** The model predicts +1 h (`bg+1:00`), yet the feature
   `[5]` is commented "prediction horizon (5min)" (it is actually the `bg_gap` sampling
   cadence), the fallback extrapolates +2 h (`slope*120`), and variables are named
   `pred_bg30`/`pred_bg60`. Pin down a single, documented horizon.

6. **The rule-based filter, not the learned policy, drives the gains.** TIR jumps 23%→61%
   (SAC) and 23%→61% (PPO) only when `SafetyFilter` is added, and `env_factory` *always*
   includes it. The ablation cannot separate "RL/LGBM" from "hand-written safety rules".
   → Add a filter-only baseline and a no-filter RL arm.

7. **"Best" model is clinically unsafe — TIR hides hypoglycemia.** The 67.9%-TIR config has
   **LBGI ≈ 28** (and SAC+Filter ≈ 52); clinically LBGI>5 is high hypo risk. The filter trades
   hyperglycemia for frequent/severe **hypoglycemia**, which is acutely more dangerous. TIR
   alone is a misleading headline. → Report LBGI/HBGI/time-below-54 with equal prominence and
   weight hypo events accordingly.

8. **Single patient, single scenario.** Everything is `adolescent#002`. No evaluation across
   simglucose's 30 virtual patients or randomized meal scenarios → no generalization evidence.

9. **Doc/code drift in the reward.** README/constants say target = 125 mg/dL (line 65), but the
   wrapper calls `continuous_reward(..., target=140.0, sigma=30.0)` (line 184); the docstring's
   "peaks at +0.648 / equals 0 at ±σ" disagrees with the `expo+1` implementation.

### C. Engineering / reproducibility / hygiene

10. **No reproducible Stage-1 → checkpoint path.** README says to copy from `model/...` but
    `model/` is git-ignored; only the pre-built `lgbm_model.pkl` survives a clone. The exact
    params that produced it aren't pinned to the artifact.
11. **Repo bloat / duplication.** `lgbm_model.pkl` is committed **twice** (root + `Result/TreeModel/`,
    13.8 MB each); TensorBoard event files, the PPO zip, and a 548 KB notebook with embedded
    images are all in-tree. Consider Git LFS / releases.
12. **Dead & duplicated code.** `train.py` lines 267–273 build local `columns_to_remove`/
    `target_encoders` that are unused (the imported `params` versions are passed instead).
    `cross_validator.py` has two near-identical classes; `remaining_indices` is computed but unused.
13. **Fragility & resource leaks.** `utils.py` import-time `open()` is never closed and is
    CWD-dependent (§6); fixes: lazy-load config, use a context manager, resolve paths from
    `__file__`.
14. **No tests / CI / type-checking / license.** Add at least smoke tests for the wrapper stack
    and the feature-row shape expected by `lgbm_model.pkl`'s transformer.
15. **Naming/typos.** `.gitignore` `__pycahce__`; `data/raw/READM.md`; mixed `gym`+`gymnasium`.

### Suggested priority order

1. Fix #1 (pred key) and #2 (normalization) — they change the numbers.
2. Re-run evaluation with multiple seeds/patients and report LBGI alongside TIR (#3, #7, #8).
3. Add a filter-only ablation to attribute the gains honestly (#6).
4. Decide whether the real-data LightGBM is defensible in-sim, or retrain a predictor on
   simglucose data (#4) — aligns with the "Future Work: simulator-pretrained predictor".
5. Clean up reproducibility/hygiene (#10–#15).
```

---

## 8. Applied fixes & verification — branch `claude/bugfix-smoke`

§7 issues #1–#3, #5, #6, #9 were confirmed and fixed in `simglucose_ppo_lgbm.py`, then
verified on a Linux box (newport, env `hykim_ect2`, **CPU** torch — the workload is
CPU/env-bound so the GPU gives no benefit).

### Confirmed (was "suspected" in §7)
- **Feature time-reversal + duplicate.** Loading the saved pipeline shows
  `feature_names_in_` expects **newest-first** (`bg_0_lag`=now … `bg_60_lag`=60 min ago; same
  for insulin/carbs). The wrapper fed **oldest-first** with a duplicated "now" and a missing
  60-min point — so the *used* features (`bg_0..30_lag`, all insulin/carbs) were corrupted.

### Fixes
- `LGBMRewardWrapper._build_feature_row`: histories reversed to newest-first, bg diffs
  recomputed (newer−older), duplicate "now" removed.
- `FeatureObsWrapper`: reads `pred_bg60` (was the never-set `pred_bg30` → the policy never
  actually saw the forecast).
- `evaluate_agent`: normalises obs with the training-time VecNormalize stats; uses a distinct
  seed per episode; reports TIR±std and time-below-54.
- `SafetyFilter.action` returns shape `(1,)`; reward target/σ are explicit constants (140/30).
- Toggles: `SIMGLU_FIX` (0 = original buggy path, kept for A/B), `SIMGLU_TAG`, `SIMGLU_STEPS`,
  `SIMGLU_ENVS`.

### Evidence #1 — predictor quality (decisive, no training)
1 h LightGBM forecast vs the simulator's *actual* BG 12 steps later, same trajectory:

| feature row | 1 h MAE | corr(pred, actual) |
|---|---:|---:|
| BUGGY (reversed) | 41.0 mg/dL | +0.858 |
| **FIXED** | **28.4 mg/dL** | **+0.961** |

→ forecast error −31 %; the reward signal the PPO agent optimises is materially more accurate.

### Evidence #2 — RL smoke A/B (15 000 steps, CPU, 5 seeds) — *pipeline + eval validity*
| | FIXED | BUGGY |
|---|---:|---:|
| TIR | 25.3 % ± 5.18 | 22.1 % ± 6.32 |
| HBGI | 33.4 | 36.8 |
| LBGI / TBR<54 | 0.00 / 0.0 % | 0.00 / 0.0 % |

The eval now shows a **real across-seed spread** (was `±0.00` = effectively n=1). At 15 k steps
both models are heavily undertrained (full run = 2.3 M = 150×), so the TIR gap is within noise
— *not* a generalisation claim. Env ≈ 15 steps/s (each step runs a full LightGBM predict +
simglucose ODE), so a full retrain is ~tens of hours on CPU; use `SubprocVecEnv` or a lighter
predictor to speed it up.
