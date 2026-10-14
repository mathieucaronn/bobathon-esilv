"""
Day 2 — Step 1: Ridge vs dummy (random row holdout, splitter=0.2)

Pushes a ComparisonReport to Skore Hub under key '02_ridge' and writes
submissions/02_ridge.csv for Kaggle.
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ──────────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")

# Same eight numeric features as Day 1 (Ridge cannot handle strings or NaNs
# without preprocessing; we add the imputer to handle missing values).
feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[feature_cols]
y = visits["target"]

# ── 3. Define models ─────────────────────────────────────────────────────────
# Dummy: always predicts the training mean — the floor to beat.
dummy = DummyRegressor(strategy="mean")

# Ridge: median-impute NaNs first, then fit a regularised linear model.
# alpha=1.0 is a mild shrinkage; larger alpha → smaller coefficients.
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# ── 4. Evaluate each model on the same random 20 % row holdout ──────────────
# splitter=0.2 is a random visit holdout (same as Day 1 dummy), so the scores
# are directly comparable. Note: this split can place visits from the same
# patient on both sides; that leakage is addressed in the next step (grouped CV).
#
# project.put() only accepts EstimatorReport or CrossValidationReport, so we
# evaluate each model individually and push both reports separately.
report_dummy = evaluate(dummy, X, y, splitter=0.2)
report_ridge = evaluate(ridge, X, y, splitter=0.2)

rmse_dummy = report_dummy.metrics.rmse()
rmse_ridge = report_ridge.metrics.rmse()
pct_improvement = (rmse_dummy - rmse_ridge) / rmse_dummy * 100

print(f"RMSE dummy : {rmse_dummy:.3f}")
print(f"RMSE ridge : {rmse_ridge:.3f}")
print(f"Improvement: {pct_improvement:+.1f}%  (positive = Ridge beats dummy)")

# ── 5. Push to Skore Hub ─────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy_v2", report_dummy)
project.put("02_ridge", report_ridge)
print("Reports pushed → check the URLs printed above")

# ── 6. Kaggle submission ─────────────────────────────────────────────────────
# Refit Ridge on ALL training visits (no CV split) — the full data gives the
# best generalising weights for the test set.
final = clone(ridge).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])

Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/02_ridge.csv", index=False)
print(f"Submission written: submissions/02_ridge.csv  shape={submission.shape}")
