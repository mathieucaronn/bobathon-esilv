"""
Day 2 — Step 7: fit the chosen model (step 5 tabular_pipeline) on all training
data and produce the Kaggle submission.

- Best model: tabular_pipeline("regressor") on X_full (no Index/patient_id).
- EstimatorReport with default splitter pushed as 07_tabular.
- Refit on all training visits → predict X_test → submissions/07_tabular.csv.
- Wrap-up workbook analysis: biggest-error visits, step-by-step gain summary,
  and the modelling problems still open.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from skore import Project, evaluate, login
from skrub import tabular_pipeline

from cv_splits import load_visits, make_grouped_cv_splits

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ──────────────────────────────────────────────────────────────────
visits  = load_visits()
X_test  = pd.read_csv("data/X_test.csv")
cv_splits = make_grouped_cv_splits(visits)

X_full = visits.drop(columns=["Index", "patient_id", "target"])
y      = visits["target"]

# ── 3. EstimatorReport (default splitter=0.2) → hub key 07_tabular ───────────
model  = tabular_pipeline("regressor")
print("Evaluating tabular_pipeline (default splitter=0.2)…")
report = evaluate(model, X_full, y)

login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("07_tabular", report)
print("Pushed 07_tabular")

# ── 4. Train vs test RMSE (overfitting check) ─────────────────────────────────
# EstimatorReport.metrics.rmse() returns a plain float (no DataFrame).
test_val  = float(report.metrics.rmse())
train_val = float(report.metrics.rmse(data_source="train"))

print()
print("── Overfitting check (07_tabular, default 0.2 holdout) ──────────────────")
print(f"  Train RMSE : {train_val:.3f}")
print(f"  Test  RMSE : {test_val:.3f}")
print(f"  Gap        : {test_val - train_val:+.3f}")

# ── 5. Refit on all training data ─────────────────────────────────────────────
final = clone(model).fit(X_full, y)

# ── 6. Predict X_test and write submission ────────────────────────────────────
# Drop the same identifier columns as training so column sets match.
X_test_features = X_test.drop(columns=["Index", "patient_id"], errors="ignore")
# Keep only the columns seen during training, in the same order.
X_test_features = X_test_features[X_full.columns]

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test_features)

Path("submissions").mkdir(exist_ok=True)
sub_path = "submissions/07_tabular.csv"
submission.to_csv(sub_path, index=False)
print(f"\nSubmission written: {sub_path}  shape={submission.shape}")

# ── 7. Wrap-up Q1: step-by-step RMSE gains ────────────────────────────────────
# All values are from grouped-CV reports (honest patient holdout).
gains = [
    ("dummy (baseline)",         16.499, None),
    ("→ Ridge alpha=0.1",        10.437, 16.499),
    ("→ HGBR (NaN as signal)",    7.434, 10.437),
    ("→ tabular_pipeline +cats",  7.430,  7.434),
    ("→ DataOps (incl. pid)",      7.573,  7.430),   # worse: identity leak
]

print()
print("── Wrap-up Q1: RMSE at each step (GroupKFold, honest) ───────────────────")
print(f"{'Step':<40} {'RMSE':>7}  {'Δ vs prev':>10}")
print("-" * 62)
for label, rmse, prev in gains:
    if prev is None:
        print(f"  {label:<38} {rmse:>7.3f}  {'—':>10}")
    else:
        delta = rmse - prev
        print(f"  {label:<38} {rmse:>7.3f}  {delta:>+10.3f}")

print()
print("  Biggest single gain: dummy → Ridge    (+6.062 RMSE, −37 %)")
print("  Second biggest:      Ridge → HGBR     (+3.003 RMSE, −29 %)")
print("  No gain:             tabular_pipeline ≈ HGBR (Δ=−0.004, within noise)")

# ── 8. Wrap-up Q2: largest-error visits (from step-5 CV predictions) ─────────
# Re-run step-5 CV to get per-visit residuals (same cv_splits).
print()
print("── Wrap-up Q2: largest-error visits (tabular step 5, CV predictions) ────")

r_cv = evaluate(tabular_pipeline("regressor"), X_full, y, splitter=cv_splits)
pe   = r_cv.metrics.prediction_error(subsample=None)
err_df = pe.frame()   # columns: split, y_true, y_pred, residuals

# Reconstruct the original visit rows for each fold.
fold_rows = []
for fold_idx, (_, test_idx) in enumerate(cv_splits):
    fold_visits = visits.iloc[test_idx].copy().reset_index(drop=True)
    fold_err    = err_df[err_df["split"] == fold_idx].reset_index(drop=True)
    fold_visits["abs_error"]  = fold_err["residuals"].abs()
    fold_visits["y_pred"]     = fold_err["y_pred"]
    fold_visits["y_true"]     = fold_err["y_true"]
    fold_rows.append(fold_visits)

all_preds = pd.concat(fold_rows, ignore_index=True)

# Top-5 % worst errors
thresh     = all_preds["abs_error"].quantile(0.95)
worst      = all_preds[all_preds["abs_error"] >= thresh].copy()

print(f"  Total CV test visits: {len(all_preds):,}")
print(f"  Worst 5 % threshold (|error| ≥ {thresh:.2f}): {len(worst):,} visits")
print()

# Compare worst vs rest on key clinical variables
rest = all_preds[all_preds["abs_error"] < thresh]

compare_cols = ["off", "time_since_intake_off", "time_since_intake_on",
                "age_at_diagnosis", "age", "ledd"]

print(f"  {'Feature':<26} {'Worst 5%':>10} {'Rest':>10}  {'diff':>8}")
print("  " + "-" * 58)
for col in compare_cols:
    w_mean = worst[col].mean()
    r_mean = rest[col].mean()
    diff   = w_mean - r_mean
    pct_nan_w = worst[col].isna().mean() * 100
    pct_nan_r = rest[col].isna().mean()  * 100
    note = f"  NaN: {pct_nan_w:.0f}% vs {pct_nan_r:.0f}%" if pct_nan_w > pct_nan_r + 5 else ""
    print(f"  {col:<26} {w_mean:>10.2f} {r_mean:>10.2f}  {diff:>+8.2f}{note}")

# Cohort breakdown
print()
print("  Cohort breakdown (worst 5 % vs all):")
for cohort in sorted(visits["cohort"].dropna().unique()):
    pct_all   = (all_preds["cohort"] == cohort).mean() * 100
    pct_worst = (worst["cohort"]     == cohort).mean() * 100
    print(f"    {cohort}: {pct_worst:.1f} % of worst vs {pct_all:.1f} % of all")

# ── 9. Wrap-up Q3: open modelling problems ────────────────────────────────────
print()
print("── Wrap-up Q3: what's not addressed yet (from docs/CONTEXT.md) ──────────")
print("  Open problems:")
print("  1. Temporal progression — no feature captures how scores change over")
print("     visits for a patient (lag features, slope, time since diagnosis).")
print("  2. Drug timing — time_since_intake_on/off is in X but we haven't")
print("     engineered a levodopa concentration curve feature.")
print("  3. Missingness structure — NaN is used by HGBR as a binary signal but")
print("     we haven't tested an explicit 'was_off_measured' indicator column.")

# ── Réponses workbook — Day 2, section 7 (bilan) ────────────────────────────
# Q1. Le plus gros gain vient du passage dummy → Ridge (16,50 → 10,44, −37 %),
#     puis de HGBR qui garde les NaN (10,44 → 7,43, −29 %). Ajouter gene et cohort
#     n'apporte rien (−0,004, dans le bruit). Garder Index et patient_id dégrade
#     le score (+0,14) et double l'écart train/test.
# Q2. Les 5 % de visites les plus mal prédites n'ont souvent pas de score off
#     (58 % contre 42 % pour les autres) : sans l'examen OFF, le modèle perd son
#     meilleur signal. Elles sont aussi un peu plus souvent en cohorte A (92,5 %
#     contre 88,9 %) et chez des patients un peu plus âgés (+1,9 an).
# Q3. La progression dans le temps n'est pas traitée : chaque visite est vue
#     seule. Prochaine variable : les scores off et on MESURÉS aux autres visites
#     du même patient (visite précédente, moyenne, pente) et le temps écoulé.
#     Jamais le target des autres visites : il n'existe pas dans le test.
# Surapprentissage : train 6,86 contre test 7,30 (écart +0,43), raisonnable.
