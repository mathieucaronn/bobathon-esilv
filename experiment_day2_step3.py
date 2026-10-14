"""
Day 2 — Step 3: patient-grouped cross-validation (GroupKFold, 5 folds).

Evaluates dummy and Ridge(alpha=0.1) with cv_splits shared across steps 3-7,
pushes CrossValidationReports to the hub, prints a comparison table, and
verifies that no patient appears on both sides of any fold.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login

from cv_splits import FEATURE_COLS, load_visits, make_grouped_cv_splits

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ──────────────────────────────────────────────────────────────────
X_test = pd.read_csv("data/X_test.csv")
visits = load_visits()

X = visits[FEATURE_COLS]
y = visits["target"]

# Precompute grouped CV splits once — reused by steps 4-7 via cv_splits.py
cv_splits = make_grouped_cv_splits(visits)

# ── 3. Models ────────────────────────────────────────────────────────────────
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=0.1),  # best alpha from step 2
)

# ── 4. Evaluate with grouped CV ──────────────────────────────────────────────
print("Evaluating dummy (grouped CV)…")
report_dummy_gcv = evaluate(dummy, X, y, splitter=cv_splits)

print("Evaluating Ridge(alpha=0.1) (grouped CV)…")
report_ridge_gcv = evaluate(ridge, X, y, splitter=cv_splits)

# ── 5. Push to hub ───────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

project.put("03_dummy_gcv", report_dummy_gcv)
url_dummy = "https://skore.probabl.ai/M&C/bobathon-esilv/estimators/<see above>"
print("Pushed 03_dummy_gcv")

project.put("03_ridge_gcv", report_ridge_gcv)
print("Pushed 03_ridge_gcv")

# ── 6. Extract mean±std RMSE from CrossValidationReport ─────────────────────
def fold_rmse_stats(report):
    """Return (mean, std) of per-fold RMSE from a CrossValidationReport.

    report.metrics.rmse() returns a DataFrame with MultiIndex columns:
      level 0 = Estimator name, level 1 = Aggregate ('mean' | 'std')
    """
    df = report.metrics.rmse()
    # Flatten: first estimator column, then mean and std aggregates
    estimator = df.columns.get_level_values(0)[0]
    mean_val = float(df[estimator]["mean"].iloc[0])
    std_val = float(df[estimator]["std"].iloc[0])
    return mean_val, std_val

dummy_mean_gcv, dummy_std_gcv = fold_rmse_stats(report_dummy_gcv)
ridge_mean_gcv, ridge_std_gcv = fold_rmse_stats(report_ridge_gcv)

# Row-holdout RMSEs from steps 1-2
RMSE_DUMMY_RAND = 16.478
RMSE_RIDGE_RAND = 10.387

# ── 7. Comparison table ──────────────────────────────────────────────────────
print()
print(f"{'Model':<28}  {'RMSE random split':>18}  {'RMSE grouped CV (mean±std)':>26}")
print("-" * 78)
print(f"{'dummy':<28}  {RMSE_DUMMY_RAND:>18.3f}  {dummy_mean_gcv:>12.3f} ± {dummy_std_gcv:<10.3f}")
print(f"{'Ridge alpha=0.1':<28}  {RMSE_RIDGE_RAND:>18.3f}  {ridge_mean_gcv:>12.3f} ± {ridge_std_gcv:<10.3f}")

# ── 8. Leakage check ─────────────────────────────────────────────────────────
print()
patient_ids = visits["patient_id"].values
leakage_found = False
for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
    train_patients = set(patient_ids[train_idx])
    test_patients = set(patient_ids[test_idx])
    overlap = train_patients & test_patients
    if overlap:
        print(f"  Fold {fold_idx}: LEAKAGE — {len(overlap)} patient(s) on both sides!")
        leakage_found = True
    else:
        print(f"  Fold {fold_idx}: OK — no patient appears in both train and test")

if not leakage_found:
    print("✓ Leakage check passed: no patient_id appears in both train and test in any fold.")
else:
    print("✗ Leakage check FAILED.")

# ── Réponses workbook — Day 2, section 3 (validation croisée par patient) ───
# Q1. Dummy : 16,48 (aléatoire) → 16,50 ± 0,34 (GroupKFold). Ridge : 10,39 → 10,44 ± 0,20.
#     Ridge change un peu plus (+0,05), mais les deux bougent peu : un modèle
#     linéaire à 8 variables ne mémorise presque pas les patients.
# Q2. Pour 10_env_stack : GroupKFold 3,369, découpage aléatoire 3,074, Kaggle public
#     3,339 et privé 3,192. Kaggle est très proche de la GroupKFold et loin du
#     découpage aléatoire, trop optimiste (même patient des deux côtés). Kaggle teste
#     sur des patients nouveaux : c'est la GroupKFold qu'il faut suivre.
