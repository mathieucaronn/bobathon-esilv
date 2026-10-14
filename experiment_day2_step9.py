"""
Day 2 — Step 9: two-stage stacking + per-patient smoothing.

Compares two stage-2 settings:
  a) same regularisation as stage 1 (min_samples_leaf=40, l2=1.0)
  b) more regularised (min_samples_leaf=80, l2=3.0)

Pushes the better variant as 09_stack_gcv.
Prints: per-fold RMSE vs 08 (3.740), mean±std, overfitting check,
        and the gain from smoothing alone.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

from cv_splits import load_visits, make_grouped_cv_splits
from features import (
    TwoStageRegressor,
    _make_hgbr,
    _s1_extra_features,
    _safe_polyfit,
    _poly_eval,
    make_features,
)

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Data ───────────────────────────────────────────────────────────────────
visits    = load_visits()
cv_splits = make_grouped_cv_splits(visits)

X_raw  = visits.drop(columns=["target"])
y      = visits["target"].values

X_feat = make_features(X_raw)
# Attach patient_id as a helper column (used only for groupby, never as feature)
X_in   = X_feat.copy()
X_in["patient_id"] = visits["patient_id"].values

# ── 3. Manual CV loop to capture smoothing gain ───────────────────────────────
# skore's evaluate clips us to the final predict output. To measure the
# gain from smoothing, we run the folds manually: for each fold we record
# stage-2 raw predictions (before smoothing) and smoothed predictions.

PREV_MEAN = 3.740    # 08_patient_feats_gcv grouped-CV RMSE

def _run_cv(
    s2_leaf: int,
    s2_l2: float,
    label: str,
) -> dict:
    """
    Run the full 5-fold CV manually for TwoStageRegressor.

    Returns a dict with fold RMSEs (raw and smoothed) and overall metrics.
    """
    model = TwoStageRegressor(
        s2_min_samples_leaf=s2_leaf,
        s2_l2_regularization=s2_l2,
    )

    fold_rmse_raw      = []
    fold_rmse_smoothed = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv_splits):
        m = clone(model)
        m.fit(X_in.iloc[tr_idx], y[tr_idx])

        X_te = X_in.iloc[te_idx].copy()
        pid  = X_te["patient_id"].values
        t_fst = X_te["t_first"].values.astype(float)
        X_feat_te = X_te[m.feat_cols_]

        # Stage-1 preds
        s1_pred = m.stage1_.predict(X_feat_te)

        # Stage-2 raw (before smoothing, before clip)
        s1_feats = _s1_extra_features(s1_pred, t_fst, pid)
        X_s2 = pd.concat(
            [X_feat_te.reset_index(drop=True), s1_feats.reset_index(drop=True)],
            axis=1,
        )[m.s2_cols_]
        raw_pred = m.stage2_.predict(X_s2)

        # Smoothed (same logic as predict)
        smoothed = raw_pred.copy()
        df_sm = pd.DataFrame({"patient_id": pid, "t_first": t_fst, "raw": raw_pred})
        for p_id, grp in df_sm.groupby("patient_id"):
            if len(grp) < m.smooth_min_visits:
                continue
            idx  = grp.index
            t    = grp["t_first"].values.astype(float)
            v    = grp["raw"].values.astype(float)
            coef = _safe_polyfit(t, v, deg=2)
            if coef is not None:
                smoothed[idx] = _poly_eval(coef, t)
        smoothed = np.clip(smoothed, 0.0, 132.0)

        y_te = y[te_idx]
        rmse_raw      = np.sqrt(((raw_pred  - y_te) ** 2).mean())
        rmse_smoothed = np.sqrt(((smoothed   - y_te) ** 2).mean())
        fold_rmse_raw.append(rmse_raw)
        fold_rmse_smoothed.append(rmse_smoothed)

        print(f"  [{label}] Fold {fold_idx}: raw={rmse_raw:.3f}  smoothed={rmse_smoothed:.3f}")

    return {
        "label":          label,
        "fold_raw":       fold_rmse_raw,
        "fold_smoothed":  fold_rmse_smoothed,
        "mean_raw":       float(np.mean(fold_rmse_raw)),
        "std_raw":        float(np.std(fold_rmse_raw)),
        "mean_smoothed":  float(np.mean(fold_rmse_smoothed)),
        "std_smoothed":   float(np.std(fold_rmse_smoothed)),
    }


print("── Variant a: stage-2 same as stage-1 (leaf=40, l2=1.0) ────────────────")
res_a = _run_cv(s2_leaf=40,  s2_l2=1.0, label="a")

print()
print("── Variant b: stage-2 more regularised (leaf=80, l2=3.0) ───────────────")
res_b = _run_cv(s2_leaf=80, s2_l2=3.0, label="b")

# ── 4. Comparison table ───────────────────────────────────────────────────────
print()
print("── Summary table ────────────────────────────────────────────────────────")
print(f"{'':4} {'Fold':>6}  {'a raw':>8} {'a smooth':>10}  {'b raw':>8} {'b smooth':>10}")
print("  " + "-" * 56)
for i in range(5):
    print(f"    {i:>6}  {res_a['fold_raw'][i]:>8.3f} {res_a['fold_smoothed'][i]:>10.3f}"
          f"  {res_b['fold_raw'][i]:>8.3f} {res_b['fold_smoothed'][i]:>10.3f}")
print(f"    {'mean':>6}  {res_a['mean_raw']:>8.3f} {res_a['mean_smoothed']:>10.3f}"
      f"  {res_b['mean_raw']:>8.3f} {res_b['mean_smoothed']:>10.3f}")
print(f"    {'std':>6}  {res_a['std_raw']:>8.3f} {res_a['std_smoothed']:>10.3f}"
      f"  {res_b['std_raw']:>8.3f} {res_b['std_smoothed']:>10.3f}")

print()
print(f"  Previous best (08): {PREV_MEAN:.3f}")
print(f"  Variant a smoothed gain vs 08: {PREV_MEAN - res_a['mean_smoothed']:+.4f}")
print(f"  Variant b smoothed gain vs 08: {PREV_MEAN - res_b['mean_smoothed']:+.4f}")
print(f"  Smoothing gain (a): raw→smoothed {res_a['mean_raw'] - res_a['mean_smoothed']:+.4f}")
print(f"  Smoothing gain (b): raw→smoothed {res_b['mean_raw'] - res_b['mean_smoothed']:+.4f}")

# ── 5. Pick best variant and compute train RMSE ───────────────────────────────
if res_a["mean_smoothed"] <= res_b["mean_smoothed"]:
    best_res    = res_a
    best_leaf   = 40
    best_l2     = 1.0
    best_label  = "a (leaf=40, l2=1.0)"
else:
    best_res    = res_b
    best_leaf   = 80
    best_l2     = 3.0
    best_label  = "b (leaf=80, l2=3.0)"

print(f"\n  Best variant: {best_label}")

# Train RMSE: fit the best model on all data, predict on all data
best_model = TwoStageRegressor(
    s2_min_samples_leaf=best_leaf,
    s2_l2_regularization=best_l2,
)
best_model.fit(X_in, y)
train_preds = best_model.predict(X_in)
train_rmse  = np.sqrt(((train_preds - y) ** 2).mean())

print()
print("── Overfitting check (best variant) ─────────────────────────────────────")
print(f"  Train RMSE : {train_rmse:.3f}")
print(f"  Test  RMSE : {best_res['mean_smoothed']:.3f}  (mean over 5 folds)")
print(f"  Gap        : {best_res['mean_smoothed'] - train_rmse:+.3f}")

# ── 6. Push best variant via skore.evaluate ───────────────────────────────────
print()
print("Evaluating best variant with skore (for hub push)…")
best_for_push = TwoStageRegressor(
    s2_min_samples_leaf=best_leaf,
    s2_l2_regularization=best_l2,
)
report = evaluate(best_for_push, X_in, y, splitter=cv_splits)

login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("09_stack_gcv", report)
print("Pushed 09_stack_gcv")

# ── Réponses workbook ─────────────────────────────────────────────────────────
print()
print("── Réponses (commentaires) ──────────────────────────────────────────────")
print(f"# Meilleure variante : {best_label}")
print(f"# RMSE (CV groupé)   : {best_res['mean_smoothed']:.3f} ± {best_res['std_smoothed']:.3f}")
print(f"# Gain vs 08         : {PREV_MEAN - best_res['mean_smoothed']:+.4f} RMSE"
      f" ({(PREV_MEAN - best_res['mean_smoothed'])/PREV_MEAN*100:+.1f} %)")
print(f"# Gain du lissage    : {best_res['mean_raw'] - best_res['mean_smoothed']:+.4f} RMSE")
print("#")
print("# Stage 2 capte les résidus systématiques de stage 1 en les représentant")
print("# comme des features per-patient (s1_pmean, s1_lin, s1_quad).")
print("# Le lissage quadratique supprime ensuite le bruit fold-à-fold en forçant")
print("# les prédictions d'un même patient à suivre une courbe régulière vs l'âge.")
