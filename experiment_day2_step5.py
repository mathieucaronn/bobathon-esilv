"""
Day 2 — Step 5: skrub tabular_pipeline with categorical columns.

- X_full drops Index, patient_id, target — keeps cohort and gene as strings.
- tabular_pipeline("regressor") wraps TableVectorizer + HGBR.
- Same GroupKFold cv_splits from cv_splits.py.
- Pushes CrossValidationReport to hub as 05_tabular_gcv.
"""

import json
import os
from pathlib import Path

import pandas as pd
from skore import Project, evaluate, login
from skrub import tabular_pipeline

from cv_splits import load_visits, make_grouped_cv_splits

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ──────────────────────────────────────────────────────────────────
visits = load_visits()
# Drop identifiers and target; keep all clinical columns including categoricals.
X_full = visits.drop(columns=["Index", "patient_id", "target"])
y = visits["target"]
cv_splits = make_grouped_cv_splits(visits)

# ── 3. Model ──────────────────────────────────────────────────────────────────
# tabular_pipeline("regressor") = TableVectorizer → HistGradientBoostingRegressor.
# TableVectorizer auto-detects dtypes: numerics pass through, low-cardinality
# strings (cohort, gene) → ToCategorical, high-cardinality → StringEncoder.
model = tabular_pipeline("regressor")

# ── 4. Evaluate with grouped CV ──────────────────────────────────────────────
print("Evaluating tabular_pipeline (grouped CV)…")
report = evaluate(model, X_full, y, splitter=cv_splits)

# ── 5. Push to hub ───────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("05_tabular_gcv", report)
print("Pushed 05_tabular_gcv")

# ── 6. Per-fold RMSE and mean±std (workbook Q1) ───────────────────────────────
# Hardcoded HGBR step-4 values for comparison (same cv_splits).
HGBR_FOLD_RMSE = [7.486, 7.580, 7.264, 7.310, 7.533]
HGBR_MEAN      = 7.434
HGBR_STD       = 0.125

fold_df   = report.metrics.rmse(aggregate=None)
est       = fold_df.columns.get_level_values(0)[0]
tab_folds = fold_df.iloc[0].values          # shape (5,)

summary   = report.metrics.rmse()
tab_mean  = float(summary[est]["mean"].iloc[0])
tab_std   = float(summary[est]["std"].iloc[0])

print()
print("── Workbook Q1: per-fold RMSE — tabular_pipeline vs 04_hgbr_gcv ────────")
print(f"{'Fold':<8} {'HGBR (step 4)':>14} {'tabular (step 5)':>17}  {'tab wins?':>10}")
print("-" * 56)
for i, (h, t) in enumerate(zip(HGBR_FOLD_RMSE, tab_folds)):
    wins = "✓" if t < h else "✗"
    print(f"  {i:<6} {h:>14.3f} {t:>17.3f}  {wins:>10}")
print(f"  {'mean':<6} {HGBR_MEAN:>14.3f} {tab_mean:>17.3f}")
print(f"  {'std':<6} {HGBR_STD:>14.3f} {tab_std:>17.3f}")

delta = HGBR_MEAN - tab_mean
print(f"\n  RMSE delta (HGBR − tabular): {delta:+.4f}  (fold std: ±{HGBR_STD:.3f})")
if abs(delta) < HGBR_STD:
    print("  → delta < fold std: improvement is within noise, not conclusive.")
else:
    print("  → delta > fold std: meaningful improvement.")

# ── 7. TableVectorizer encoder choices (workbook Q2) ─────────────────────────
# Fit once on all data to inspect the fitted transformers.
model_fit = tabular_pipeline("regressor")
model_fit.fit(X_full, y)
tv = model_fit.named_steps["tablevectorizer"]

print()
print("── Workbook Q2: TableVectorizer encoder per column ──────────────────────")
print(f"{'Column':<26} {'Encoder':<22} {'Distinct values':>16}")
print("-" * 66)
for col, enc in tv.transformers_.items():
    n_unique = int(X_full[col].nunique())
    print(f"  {col:<24} {type(enc).__name__:<22} {n_unique:>16}")

# ── 8. Overfitting check ──────────────────────────────────────────────────────
train_df   = report.metrics.rmse(data_source="train")
train_mean = float(train_df[est]["mean"].iloc[0])
gap        = tab_mean - train_mean

print()
print("── Overfitting check: train RMSE vs test RMSE ───────────────────────────")
print(f"  Train RMSE (mean): {train_mean:.3f}")
print(f"  Test  RMSE (mean): {tab_mean:.3f}")
print(f"  Gap (test − train): {gap:+.3f}")

# ── Réponses workbook — Day 2, section 5 (skrub tabular_pipeline) ──────────
# Q1. Non : 7,430 ± 0,122 contre 7,434 ± 0,125 pour HGBR, soit 0,004 de gain,
#     30 fois moins que l'écart entre folds. Le modèle ne gagne que sur 3 folds
#     sur 5 : gene et cohort n'apportent presque rien de plus, c'est du bruit.
# Q2. Les 8 colonnes numériques passent telles quelles (PassThrough). cohort
#     (2 valeurs) et gene (4 valeurs) reçoivent ToCategorical : peu de valeurs,
#     donc HGBR les lit comme des catégories. Aucune colonne n'a assez de valeurs
#     pour justifier un StringEncoder : le choix colle bien à la cardinalité.
# Surapprentissage : train 6,85 contre test 7,43 (écart +0,58, contre +0,53 à
#     l'étape 4). Les deux colonnes en plus font mémoriser un peu plus, sans gain.
