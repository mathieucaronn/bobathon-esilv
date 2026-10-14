"""
Day 2 — Step 4: HistGradientBoostingRegressor, missing values kept as signal.

- No imputer: HGBR handles NaNs natively (a missing OFF can itself be a signal).
- Same FEATURE_COLS and GroupKFold cv_splits from cv_splits.py as step 3.
- Pushes CrossValidationReport to hub as 04_hgbr_gcv.
- Prints workbook tables: per-fold RMSE, fit/predict times, permutation importance
  vs Ridge coefficients, and train vs test RMSE (overfitting check).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login

from cv_splits import FEATURE_COLS, load_visits, make_grouped_cv_splits

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ──────────────────────────────────────────────────────────────────
visits = load_visits()
X = visits[FEATURE_COLS]
y = visits["target"]
cv_splits = make_grouped_cv_splits(visits)

# ── 3. Models ────────────────────────────────────────────────────────────────
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=0.1),
)
# No imputer: HGBR routes NaN rows to the most informative branch at each split.
hgbr = HistGradientBoostingRegressor(random_state=0)

# ── 4. Evaluate with grouped CV ──────────────────────────────────────────────
print("Evaluating Ridge(alpha=0.1) (grouped CV)…")
report_ridge = evaluate(ridge, X, y, splitter=cv_splits)

print("Evaluating HGBR (grouped CV)…")
report_hgbr = evaluate(hgbr, X, y, splitter=cv_splits)

# ── 5. Push HGBR report to hub ───────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("04_hgbr_gcv", report_hgbr)
print("Pushed 04_hgbr_gcv")

# ── 6. Per-fold RMSE table (workbook Q1) ─────────────────────────────────────
ridge_folds = report_ridge.metrics.rmse(aggregate=None)
hgbr_folds  = report_hgbr.metrics.rmse(aggregate=None)

# Both DataFrames have columns = Split #0..#4 (under the estimator level).
ridge_vals = ridge_folds.iloc[0].values   # shape (5,)
hgbr_vals  = hgbr_folds.iloc[0].values

print()
print("── Workbook Q1: RMSE per fold ──────────────────────────────────────────")
print(f"{'Fold':<8} {'Ridge':>10} {'HGBR':>10}  {'HGBR wins?':>12}")
print("-" * 46)
for i, (r_val, h_val) in enumerate(zip(ridge_vals, hgbr_vals)):
    wins = "✓" if h_val < r_val else "✗"
    print(f"  {i:<6} {r_val:>10.3f} {h_val:>10.3f}  {wins:>12}")

ridge_mean = ridge_vals.mean(); ridge_std = ridge_vals.std()
hgbr_mean  = hgbr_vals.mean();  hgbr_std  = hgbr_vals.std()
print(f"  {'mean':<6} {ridge_mean:>10.3f} {hgbr_mean:>10.3f}")
print(f"  {'std':<6} {ridge_std:>10.3f} {hgbr_std:>10.3f}")

# ── 7. Fit & predict times (workbook Q2) ─────────────────────────────────────
def timing_row(report):
    fit_df  = report.metrics.fit_time()
    pred_df = report.metrics.predict_time()
    est = fit_df.columns.get_level_values(0)[0]
    fit_mean  = float(fit_df[est]["mean"].iloc[0])
    pred_mean = float(pred_df[est]["mean"].iloc[0])
    return fit_mean, pred_mean

ridge_fit, ridge_pred = timing_row(report_ridge)
hgbr_fit,  hgbr_pred  = timing_row(report_hgbr)

print()
print("── Workbook Q2: fit and predict times (mean over 5 folds) ──────────────")
print(f"{'Model':<28} {'Fit (s)':>10} {'Predict (s)':>14}")
print("-" * 56)
print(f"{'Ridge(alpha=0.1) + imputer':<28} {ridge_fit:>10.4f} {ridge_pred:>14.4f}")
print(f"{'HGBR(random_state=0)':<28} {hgbr_fit:>10.4f} {hgbr_pred:>14.4f}")
print(f"  HGBR / Ridge fit ratio: {hgbr_fit / ridge_fit:.1f}x slower")

# ── 8. Permutation importance HGBR vs Ridge coefficients (workbook Q3) ───────
# Fit each model once on all training data for inspection.
ridge_fitted = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=0.1)).fit(X, y)
hgbr_fitted  = HistGradientBoostingRegressor(random_state=0).fit(X, y)

perm = permutation_importance(
    hgbr_fitted, X, y,
    n_repeats=10,
    random_state=0,
    scoring="neg_root_mean_squared_error",
)
perm_mean = perm.importances_mean
perm_std  = perm.importances_std

# Ridge coefficients (last step of pipeline)
ridge_coef = ridge_fitted.named_steps["ridge"].coef_

# Sort by HGBR importance descending
order = np.argsort(perm_mean)[::-1]

print()
print("── Workbook Q3: feature importance comparison ───────────────────────────")
print(f"{'Feature':<26} {'HGBR perm imp (mean±std)':>26} {'Ridge coef':>12}")
print("-" * 68)
for i in order:
    col = FEATURE_COLS[i]
    print(f"  {col:<24} {perm_mean[i]:>12.4f} ± {perm_std[i]:<10.4f} {ridge_coef[i]:>12.4f}")

# ── 9. Overfitting check: train vs test RMSE (workbook bonus) ────────────────
hgbr_test_mean  = hgbr_mean   # already computed above
hgbr_train_df   = report_hgbr.metrics.rmse(data_source="train")
hgbr_est        = hgbr_train_df.columns.get_level_values(0)[0]
hgbr_train_mean = float(hgbr_train_df[hgbr_est]["mean"].iloc[0])

ridge_test_mean  = ridge_mean
ridge_train_df   = report_ridge.metrics.rmse(data_source="train")
ridge_est        = ridge_train_df.columns.get_level_values(0)[0]
ridge_train_mean = float(ridge_train_df[ridge_est]["mean"].iloc[0])

print()
print("── Overfitting check: train RMSE vs test RMSE (mean over folds) ─────────")
print(f"{'Model':<28} {'Train RMSE':>12} {'Test RMSE':>12} {'Gap':>8}")
print("-" * 64)
print(f"{'Ridge(alpha=0.1)':<28} {ridge_train_mean:>12.3f} {ridge_test_mean:>12.3f} {ridge_test_mean - ridge_train_mean:>+8.3f}")
print(f"{'HGBR(random_state=0)':<28} {hgbr_train_mean:>12.3f} {hgbr_test_mean:>12.3f} {hgbr_test_mean - hgbr_train_mean:>+8.3f}")

# ── Réponses workbook — Day 2, section 4 (HistGradientBoosting) ────────────
# Q1. HGBR bat Ridge sur les 5 folds (7,26 à 7,58 contre 10,25 à 10,76), pas seulement
#     en moyenne : 7,43 ± 0,13 contre 10,44 ± 0,18. Le gain (≈ 3 points) dépasse
#     largement la variation entre folds, donc il est réel.
# Q2. HGBR met 0,74 s à s'entraîner contre 0,02 s pour Ridge (≈ 39 fois plus lent),
#     et 0,012 s pour prédire. Ça reste rapide : le gain de RMSE vaut le surcoût.
# Q3. HGBR s'appuie surtout sur off (8,4) et on (5,6), puis age (2,2) et
#     age_at_diagnosis (1,9). Ridge donnait le plus gros coefficient à
#     time_since_intake_on (1,75), mais son importance n'est que de 0,48 : le
#     coefficient était gros à cause de l'unité (heures), pas de son utilité.
# Surapprentissage : train 6,91 contre test 7,43 (écart +0,53), contre +0,01 pour
#     Ridge. HGBR mémorise un peu, mais l'écart reste faible : pas de problème ici.
