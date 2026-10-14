"""
Experiment 10: upper-envelope features + TwoStageRegressor.

Evaluates TwoStageRegressor() with GroupKFold(5), pushes as 10_env_stack_gcv.
If CV RMSE < 3.430 (09 best): also pushes EstimatorReport as 10_env_stack and
writes submissions/10_env_stack.csv. Otherwise falls back to the 09 model.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from skore import Project, evaluate, login

from cv_splits import load_visits, make_grouped_cv_splits
from features import TwoStageRegressor, make_features

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ───────────────────────────────────────────────────────────────────
visits    = load_visits()
cv_splits = make_grouped_cv_splits(visits)
X_test    = pd.read_csv("data/X_test.csv")
sample    = pd.read_csv("data/sample_submission.csv")

X_raw  = visits.drop(columns=["target"])
y      = visits["target"]

print("Building training features (with envelope)…")
X_feat = make_features(X_raw)
X_in   = X_feat.copy()
X_in["patient_id"] = visits["patient_id"].values
feat_cols = X_in.columns.tolist()
print(f"  Shape: {X_in.shape}  ({X_in.shape[1]} cols, including patient_id helper)")

# ── 3. GroupKFold CV ──────────────────────────────────────────────────────────
PREV_BEST = 3.430
model = TwoStageRegressor()

print("Evaluating TwoStageRegressor (GroupKFold)…")
report_cv = evaluate(model, X_in, y, splitter=cv_splits)

est = report_cv.metrics.rmse().columns.get_level_values(0)[0]
cv_mean = float(report_cv.metrics.rmse()[est]["mean"].iloc[0])
cv_std  = float(report_cv.metrics.rmse()[est]["std"].iloc[0])
fold_df = report_cv.metrics.rmse(aggregate=None)
fold_vals = fold_df.iloc[0].values

train_mean = float(report_cv.metrics.rmse(data_source="train")[est]["mean"].iloc[0])

# ── 4. Push CV report ─────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("10_env_stack_gcv", report_cv)
print("Pushed 10_env_stack_gcv")

# ── 5. Print results ──────────────────────────────────────────────────────────
print()
print("── Per-fold RMSE vs 09 (3.430) ──────────────────────────────────────────")
print(f"  {'Fold':<6} {'RMSE':>8}  {'beats 09?':>10}")
for i, v in enumerate(fold_vals):
    print(f"  {i:<6} {v:>8.3f}  {'✓' if v < PREV_BEST else '✗':>10}")
print(f"  {'mean':<6} {cv_mean:>8.3f}")
print(f"  {'std':<6} {cv_std:>8.3f}")
print(f"\n  09 RMSE: {PREV_BEST:.3f}  →  Δ = {cv_mean - PREV_BEST:+.4f}")

print()
print("── Overfitting check ─────────────────────────────────────────────────────")
print(f"  Train RMSE: {train_mean:.3f}  |  Test RMSE: {cv_mean:.3f}  |  Gap: {cv_mean - train_mean:+.3f}")

# ── 6. Submission (best model) ────────────────────────────────────────────────
if cv_mean < PREV_BEST:
    sub_key   = "10_env_stack"
    sub_file  = "submissions/10_env_stack.csv"
    print(f"\n  ✓ New best ({cv_mean:.3f} < {PREV_BEST:.3f}) — submitting 10_env_stack")
else:
    sub_key   = "09_stack"
    sub_file  = "submissions/09_stack.csv"
    print(f"\n  ✗ No improvement ({cv_mean:.3f} ≥ {PREV_BEST:.3f}) — reusing 09_stack submission")

# EstimatorReport + refit only when we have a new best
if sub_key == "10_env_stack":
    print("Evaluating (EstimatorReport, splitter=0.2)…")
    report_est = evaluate(clone(model), X_in, y)
    est_test  = float(report_est.metrics.rmse())
    est_train = float(report_est.metrics.rmse(data_source="train"))
    print(f"  Train: {est_train:.3f}  Test: {est_test:.3f}  Gap: {est_test - est_train:+.3f}")
    project.put(sub_key, report_est)
    print(f"Pushed {sub_key}")

    print("Refitting on all training data…")
    final = clone(model).fit(X_in, y)

    print("Building test features…")
    X_test_feat = make_features(X_test)
    X_test_in   = X_test_feat.copy()
    X_test_in["patient_id"] = X_test["patient_id"].values
    for col in feat_cols:
        if col not in X_test_in.columns:
            X_test_in[col] = float("nan")
    X_test_in = X_test_in[feat_cols]

    preds = final.predict(X_test_in)
    submission = X_test[["Index"]].copy()
    submission["target"] = preds
    submission = submission.set_index("Index").reindex(sample["Index"]).reset_index()
    submission.columns = ["Index", "target"]

    assert len(submission) == 11013
    assert submission["target"].isna().sum() == 0
    assert submission["target"].between(0, 132).all()

    Path("submissions").mkdir(exist_ok=True)
    submission.to_csv(sub_file, index=False)
    print(f"\nSubmission: {sub_file}  shape={submission.shape}")
    print(f"  range [{submission['target'].min():.2f}, {submission['target'].max():.2f}]  "
          f"mean {submission['target'].mean():.3f}")
else:
    print(f"  Using existing {sub_file}")
    est_test = None
