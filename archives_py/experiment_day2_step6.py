"""
Day 2 — Step 6: skrub DataOps with patient-grouped CV baked into the graph.

- GroupKFold(5) and split_kwargs={'groups': patient_id} live on mark_as_X,
  so evaluate(pred, data=...) reads them without a separate cv_splits list.
- X_op is data.drop("target") — it still contains Index and patient_id.
- Pushes CrossValidationReport to hub as 06_dataops_gcv.
"""

import json
import os
from pathlib import Path

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

from cv_splits import load_visits

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ──────────────────────────────────────────────────────────────────
visits = load_visits()

# ── 3. Build the DataOps graph (exactly as in docs/DAY2.md step 6) ───────────
data   = skrub.var("visits", visits)
groups = data["patient_id"]

# mark_as_X bakes the GroupKFold + patient groups into the graph node itself.
# evaluate(pred) will read cv and split_kwargs from here — no separate cv_splits.
X_op = data.drop("target", axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()

pred = (
    X_op
    .skb.apply(skrub.TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

# ── 4. Evaluate — splitter comes from the graph, not from a separate argument ─
print("Evaluating DataOps graph (grouped CV from graph)…")
report = evaluate(pred, data={"visits": visits})

# ── 5. Push to hub ────────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("06_dataops_gcv", report)
print("Pushed 06_dataops_gcv")

# ── 6. Extract metrics ────────────────────────────────────────────────────────
est = report.metrics.rmse().columns.get_level_values(0)[0]

summary   = report.metrics.rmse()
tab_mean  = float(summary[est]["mean"].iloc[0])
tab_std   = float(summary[est]["std"].iloc[0])

fold_df   = report.metrics.rmse(aggregate=None)
tab_folds = fold_df.iloc[0].values          # shape (5,)

train_df   = report.metrics.rmse(data_source="train")
train_mean = float(train_df[est]["mean"].iloc[0])

# Step 5 reference values (same model, same cv splits, but Index/patient_id dropped)
STEP5_FOLDS = [7.475, 7.551, 7.289, 7.311, 7.524]
STEP5_MEAN  = 7.430
STEP5_STD   = 0.122

# ── 7. Workbook Q1: compare scores with step 5 ───────────────────────────────
print()
print("── Workbook Q1: per-fold RMSE — 06_dataops_gcv vs 05_tabular_gcv ───────")
print(f"{'Fold':<8} {'step 5 (no Index/pid)':>22} {'step 6 (DataOps)':>18}")
print("-" * 52)
for i, (s5, s6) in enumerate(zip(STEP5_FOLDS, tab_folds)):
    print(f"  {i:<6} {s5:>22.3f} {s6:>18.3f}")
print(f"  {'mean':<6} {STEP5_MEAN:>22.3f} {tab_mean:>18.3f}")
print(f"  {'std':<6} {STEP5_STD:>22.3f} {tab_std:>18.3f}")
delta = tab_mean - STEP5_MEAN
print(f"\n  RMSE delta (DataOps − step5): {delta:+.4f}")

# ── 8. Workbook Q2: columns in X_op vs step 5 ────────────────────────────────
X_op_cols   = list(X_op.skb.eval().columns)
step5_cols  = list(visits.drop(columns=["Index", "patient_id", "target"]).columns)
extra_cols  = [c for c in X_op_cols if c not in step5_cols]

print()
print("── Workbook Q2: columns in X_op vs step 5 X_full ────────────────────────")
print(f"  step 5 X_full columns ({len(step5_cols)}): {step5_cols}")
print(f"  X_op columns ({len(X_op_cols)}):           {X_op_cols}")
print(f"  Extra columns in X_op (not in step 5):  {extra_cols}")
print()
print("  → Index and patient_id ARE present in X_op.")
print("    If the model learns from patient_id, it memorises training patients")
print("    instead of generalising — a form of identity leakage. GroupKFold")
print("    prevents the same patient appearing on both sides, so the CV RMSE")
print("    won't look artificially good, but the Kaggle test patients are new,")
print("    so any weight on patient_id will not transfer.")

# ── 9. Overfitting check ──────────────────────────────────────────────────────
gap = tab_mean - train_mean
print()
print("── Overfitting check: train RMSE vs test RMSE ───────────────────────────")
print(f"  Train RMSE (mean): {train_mean:.3f}")
print(f"  Test  RMSE (mean): {tab_mean:.3f}")
print(f"  Gap (test − train): {gap:+.3f}")

# ── Réponses workbook — Day 2, section 6 (skrub DataOps) ───────────────────
# Q1. Non : 7,573 ± 0,147 contre 7,430 ± 0,122 à l'étape 5, soit +0,14, et c'est
#     pire sur les 5 folds. Deux raisons : X_op garde Index et patient_id, et un
#     TableVectorizer() seul encode cohort et gene en one-hot, alors que
#     tabular_pipeline les mettait en catégories (ToCategorical) pour HGBR.
# Q2. X_op a 12 colonnes : les 10 de l'étape 5 plus Index (passé tel quel) et
#     patient_id (StringEncoder). Sur un fold de test, leur importance par
#     permutation est nulle (0,013 et -0,007, contre 8,2 pour off) : sur des
#     patients nouveaux, un identifiant n'apprend rien. Mais le modèle s'en sert
#     sur le train (écart train/test +1,23 contre +0,58) : il mémorise les
#     patients, ce qui gaspille des splits et dégrade le score. Il faut les retirer.
