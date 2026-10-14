"""
Phase C: Kaggle submission for 09_stack (TwoStageRegressor, variant a).

- EstimatorReport (default splitter=0.2) pushed as 09_stack.
- Refits TwoStageRegressor on all training visits.
- Builds test features with identical column order.
- Writes submissions/09_stack.csv (11 013 rows, Index,target, no NaN, [0,132]).
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from skore import Project, evaluate, login

from cv_splits import load_visits
from features import TwoStageRegressor, make_features

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ───────────────────────────────────────────────────────────────────
visits = load_visits()
X_test  = pd.read_csv("data/X_test.csv")
sample  = pd.read_csv("data/sample_submission.csv")

X_raw = visits.drop(columns=["target"])
y     = visits["target"]

# ── 3. Build training features ────────────────────────────────────────────────
print("Building training features…")
X_feat = make_features(X_raw)
X_in   = X_feat.copy()
X_in["patient_id"] = visits["patient_id"].values
feat_cols = X_in.columns.tolist()   # canonical column order (42 feats + patient_id)
print(f"  Training matrix: {X_in.shape}")

# ── 4. Model ──────────────────────────────────────────────────────────────────
model = TwoStageRegressor()   # variant a defaults: leaf=40, l2=1.0

# ── 5. EstimatorReport (default splitter=0.2) and push ───────────────────────
print("Evaluating (EstimatorReport, splitter=0.2)…")
report = evaluate(model, X_in, y)

test_rmse  = float(report.metrics.rmse())
train_rmse = float(report.metrics.rmse(data_source="train"))
print(f"  Train RMSE: {train_rmse:.3f}  |  Test RMSE: {test_rmse:.3f}  |  Gap: {test_rmse - train_rmse:+.3f}")

login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("09_stack", report)
print("Pushed 09_stack")

# ── 6. Refit on all training visits ──────────────────────────────────────────
print("Refitting on all training visits…")
final = clone(model).fit(X_in, y)

# ── 7. Build test features with identical column order ───────────────────────
print("Building test features…")
X_test_feat = make_features(X_test)
X_test_in   = X_test_feat.copy()
X_test_in["patient_id"] = X_test["patient_id"].values

# Align to training column order exactly
for col in feat_cols:
    if col not in X_test_in.columns:
        X_test_in[col] = float("nan")
X_test_in = X_test_in[feat_cols]
print(f"  Test matrix aligned: {X_test_in.shape}")

# ── 8. Predict ────────────────────────────────────────────────────────────────
preds = final.predict(X_test_in)   # already clipped to [0, 132] by TwoStageRegressor

# ── 9. Write submission in sample_submission row order ────────────────────────
submission = X_test[["Index"]].copy()
submission["target"] = preds
submission = submission.set_index("Index").reindex(sample["Index"]).reset_index()
submission.columns = ["Index", "target"]

assert len(submission) == 11013,             f"Expected 11013 rows, got {len(submission)}"
assert submission["target"].isna().sum() == 0, "NaN predictions found!"
assert submission["target"].between(0, 132).all(), "Predictions outside [0, 132]!"

Path("submissions").mkdir(exist_ok=True)
out_path = "submissions/09_stack.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  shape={submission.shape}")
print(f"  target range : [{submission['target'].min():.2f}, {submission['target'].max():.2f}]")
print(f"  target mean  : {submission['target'].mean():.3f}")
print(f"  NaN count    : {submission['target'].isna().sum()}")
