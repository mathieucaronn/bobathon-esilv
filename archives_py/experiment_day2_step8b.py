"""
Day 2 — Step 8b: Kaggle submission for 08_patient_feats.

- Pushes an EstimatorReport (default splitter=0.2) as 08_patient_feats.
- Refits on all training visits.
- Builds make_features(X_test) with columns aligned to training.
- Writes submissions/08_patient_feats.csv in the same row order as
  data/sample_submission.csv (11 013 rows, no NaN).
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from skore import Project, evaluate, login

from cv_splits import load_visits
from features import make_features

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ───────────────────────────────────────────────────────────────────
visits  = load_visits()
X_test  = pd.read_csv("data/X_test.csv")
sample  = pd.read_csv("data/sample_submission.csv")   # canonical row order

X_raw   = visits.drop(columns=["target"])
y       = visits["target"]

# ── 3. Build features ─────────────────────────────────────────────────────────
print("Building training features…")
X_feat  = make_features(X_raw)
feat_cols = X_feat.columns.tolist()   # canonical column order
print(f"  Training feature matrix: {X_feat.shape}")

# ── 4. Model ──────────────────────────────────────────────────────────────────
hgbr = HistGradientBoostingRegressor(
    random_state=0,
    max_iter=800,
    learning_rate=0.03,
    min_samples_leaf=40,
    l2_regularization=1.0,
)

# ── 5. EstimatorReport (default splitter=0.2) and push ───────────────────────
print("Evaluating (EstimatorReport, splitter=0.2)…")
report = evaluate(hgbr, X_feat, y)

test_rmse  = float(report.metrics.rmse())
train_rmse = float(report.metrics.rmse(data_source="train"))
print(f"  Train RMSE: {train_rmse:.3f}  |  Test RMSE: {test_rmse:.3f}  |  Gap: {test_rmse - train_rmse:+.3f}")

login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("08_patient_feats", report)
print("Pushed 08_patient_feats")

# ── 6. Refit on all training visits ──────────────────────────────────────────
print("Refitting on all training visits…")
final = clone(hgbr).fit(X_feat, y)

# ── 7. Build test features with identical columns ─────────────────────────────
print("Building test features…")
X_test_feat = make_features(X_test)
print(f"  Test feature matrix raw: {X_test_feat.shape}")

# Align columns: add any training column missing in test (fill NaN), drop extras.
for col in feat_cols:
    if col not in X_test_feat.columns:
        X_test_feat[col] = float("nan")
X_test_feat = X_test_feat[feat_cols]
print(f"  Test feature matrix aligned: {X_test_feat.shape}")

# ── 8. Predict and write submission ──────────────────────────────────────────
preds = final.predict(X_test_feat)

# Align to sample_submission row order (by Index).
submission = X_test[["Index"]].copy()
submission["target"] = preds

# Reindex to match sample_submission.csv exactly.
submission = submission.set_index("Index").reindex(sample["Index"]).reset_index()
submission.columns = ["Index", "target"]

assert len(submission) == 11013, f"Expected 11013 rows, got {len(submission)}"
assert submission["target"].isna().sum() == 0, "NaN predictions found!"

Path("submissions").mkdir(exist_ok=True)
out_path = "submissions/08_patient_feats.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  shape={submission.shape}")
print(f"  target range: [{submission['target'].min():.2f}, {submission['target'].max():.2f}]")
print(f"  target mean : {submission['target'].mean():.3f}")
