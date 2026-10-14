"""
Day 2 — Step 2: Ridge alpha sweep (0.1, 1, 10, 100, 1000)

For each alpha:
  - evaluates Ridge(alpha) + median imputer on the same 8-feature / splitter=0.2 setup
  - pushes the report to Skore Hub under key 02_ridge_alpha<alpha>
  - prints a comparison table vs dummy (RMSE 16.478)

Then refits the best-alpha pipeline on all training data and writes the
Kaggle submission CSV.
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

RMSE_DUMMY = 16.478  # from experiment_day2.py

# ── 3. Login once ────────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

# ── 4. Alpha sweep ───────────────────────────────────────────────────────────
alphas = [0.1, 1, 10, 100, 1000]
results = []  # list of (alpha, key, rmse, report)

for alpha in alphas:
    key = f"02_ridge_alpha{alpha:g}"
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=alpha),
    )
    report = evaluate(pipeline, X, y, splitter=0.2)
    rmse = report.metrics.rmse()
    project.put(key, report)
    results.append((alpha, key, rmse, report))
    print(f"  alpha={alpha:>6g}  RMSE={rmse:.3f}  pushed → {key}")

# ── 5. Comparison table ──────────────────────────────────────────────────────
print()
print(f"{'Model':<30}  {'RMSE':>7}  {'vs dummy':>9}")
print("-" * 52)
print(f"{'dummy (mean)':<30}  {RMSE_DUMMY:>7.3f}  {'—':>9}")
for alpha, key, rmse, _ in results:
    pct = (RMSE_DUMMY - rmse) / RMSE_DUMMY * 100
    print(f"{'Ridge alpha=' + str(alpha):<30}  {rmse:>7.3f}  {pct:>+8.1f}%")

# ── 6. Pick best alpha ───────────────────────────────────────────────────────
best_alpha, best_key, best_rmse, _ = min(results, key=lambda t: t[2])
print()
print(f"Best alpha: {best_alpha}  (RMSE {best_rmse:.3f})")

# ── 7. Refit on all training data → Kaggle submission ────────────────────────
best_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=best_alpha),
)
final = clone(best_pipeline).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])

Path("submissions").mkdir(exist_ok=True)
out_path = f"submissions/{best_key}.csv"
submission.to_csv(out_path, index=False)
print(f"Submission written: {out_path}  shape={submission.shape}")

# ── 8. Hub URL for Kaggle description ────────────────────────────────────────
# Retrieve the URL from the last put call for the best key.
# skore prints the URL at put time; reconstruct the pattern here.
print()
print(f"Hub key to paste in Kaggle Submission Description: {best_key}")
print("(URL was printed by skore above when that report was pushed)")
