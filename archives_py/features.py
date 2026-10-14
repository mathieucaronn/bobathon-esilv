"""
Patient-trajectory feature engineering for the Parkinson's grouped-CV pipeline.

make_features(df) works identically on train visits and test visits:
  - df must contain columns: patient_id, age, age_at_diagnosis, on, off, ledd,
    time_since_intake_on, time_since_intake_off  (plus Index if present).
  - It does NOT use the 'target' column.
  - It does NOT emit patient_id or Index as model features.
  - Sort is done internally (patient_id, age); original row order is restored.

Anti-leak guarantee: all per-patient aggregates (mean, fit, lag) use ONLY the
columns available in X_test. Since test patients each have ~8 visits, statistics
computed across those visits are valid features for every visit of that patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _envelope_line(t: np.ndarray, e: np.ndarray, n_iter: int = 4, cut: float = 2.0) -> np.ndarray:
    """Line fit that drops points far below it, so it follows the top of the measurements."""
    m = np.isfinite(e) & np.isfinite(t)
    if m.sum() < 2:
        return np.full(len(t), np.nanmax(e) if m.any() else np.nan)
    tt, ee = t[m], e[m]
    keep = np.ones(len(tt), bool)
    c = np.polyfit(tt, ee, 1)          # initial fit before loop
    for _ in range(n_iter):
        if keep.sum() < 2:
            break
        c = np.polyfit(tt[keep], ee[keep], 1)
        keep = (ee - np.polyval(c, tt)) > -cut
    return np.polyval(c, t)


def _safe_polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray | None:
    """Return polyfit coefficients or None when there are fewer than deg+1 points."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < deg + 1:
        return None
    return np.polyfit(x[mask], y[mask], deg)


def _poly_eval(coeffs, x_vals: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial (coefficients from np.polyfit) at x_vals."""
    return np.polyval(coeffs, x_vals)


def _ridge_weighted_line(
    ages: np.ndarray,
    values: np.ndarray,
    prior_slope: float = 2.8,
    lam: float = 5.0,
) -> tuple[float, float]:
    """
    Per-patient ridge-regularised linear fit of `values` vs `ages`.

    Penalty pulls the slope toward `prior_slope`.  Returns (intercept, slope).
    Solves: minimise ||y - (a + b*x)||^2 + lam*(b - prior_slope)^2.

    Falls back to (mean(values), prior_slope) when fewer than 2 valid points.
    """
    mask = np.isfinite(ages) & np.isfinite(values)
    if mask.sum() < 2:
        return float(np.nanmean(values)) if mask.any() else 0.0, prior_slope

    x = ages[mask]
    y = values[mask]
    n = len(x)
    xm = x.mean()
    ym = y.mean()

    # Ridge normal equations (centre x for numerical stability)
    xc = x - xm
    # slope: (Σ xc*y + lam*prior_slope) / (Σ xc² + lam)
    slope = (xc @ y + lam * prior_slope) / (xc @ xc + lam)
    intercept = ym - slope * xm
    return float(intercept), float(slope)


# ── on_corr: correct ON score for time since intake ──────────────────────────

def _on_corrected(on: pd.Series, time_since_intake_on: pd.Series) -> pd.Series:
    """
    Approximate true-OFF equivalent from the ON score.

    ON ≈ true_OFF * response_factor(t), where:
      response_factor(t) = 0.44 + 0.30 * exp(-clip(t, 0, 6) / 0.6)
    Inverting: true_OFF_proxy ≈ ON / response_factor.

    At t=0   response≈0.74 → on_corr ≈ on / 0.74 (early peak)
    At t=1.5 response≈0.44 → on_corr ≈ on / 0.44 (worn off)
    """
    t = time_since_intake_on.clip(0, 6)
    factor = 0.44 + 0.30 * np.exp(-t / 0.6)
    return on / factor


# ── main feature builder ──────────────────────────────────────────────────────

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build patient-trajectory features from a visits DataFrame.

    Parameters
    ----------
    df : DataFrame with at minimum:
         patient_id, age, age_at_diagnosis, on, off, ledd,
         time_since_intake_on, time_since_intake_off.
         May also contain Index and target (target is ignored).

    Returns
    -------
    DataFrame of engineered features indexed like df (original row order).
    Does NOT contain patient_id, Index, or target.
    """
    df = df.copy()
    original_index = df.index

    # Sort within each patient by age so lags and fits are chronological.
    df = df.sort_values(["patient_id", "age"]).reset_index(drop=False)

    pid = df["patient_id"]

    # ── 1. Temporal position features ────────────────────────────────────────
    df["dur"]        = df["age"] - df["age_at_diagnosis"]
    t_first          = df.groupby("patient_id")["age"].transform("min")
    df["t_first"]    = df["age"] - t_first
    df["visit_rank"] = df.groupby("patient_id").cumcount()
    df["n_visits"]   = df.groupby("patient_id")["age"].transform("count")

    # ── 2. Missingness flags ──────────────────────────────────────────────────
    df["off_missing"] = df["off"].isna().astype(np.float32)
    df["on_missing"]  = df["on"].isna().astype(np.float32)

    # ── 3. ON-corrected score and ON/OFF ratio ────────────────────────────────
    df["on_corr"]      = _on_corrected(df["on"], df["time_since_intake_on"])
    df["on_off_ratio"] = df["on"] / df["off"]   # NaN when either is missing

    # ── 4. Per-patient stats for off, on, on_corr ─────────────────────────────
    signal_cols = ["off", "on", "on_corr"]

    for c in signal_cols:
        g = df.groupby("patient_id")[c]
        df[f"{c}_pmean"]  = g.transform("mean")
        df[f"{c}_pmax"]   = g.transform("max")
        df[f"{c}_pcount"] = g.transform("count")

        # Linear and quadratic fit of c vs t_first, evaluated at each visit.
        # For each patient: fit coefficients on all non-NaN visits, then evaluate.
        lin_vals  = np.full(len(df), np.nan)
        quad_vals = np.full(len(df), np.nan)

        for pat_id, grp in df.groupby("patient_id"):
            idx   = grp.index
            t     = grp["t_first"].values.astype(float)
            y_raw = grp[c].values.astype(float)

            lin_coef  = _safe_polyfit(t, y_raw, deg=1)
            quad_coef = _safe_polyfit(t, y_raw, deg=2)

            if lin_coef is not None:
                lin_vals[idx]  = _poly_eval(lin_coef, t)
            # else: leave as NaN — HGBR handles missing values natively.

            if quad_coef is not None:
                quad_vals[idx] = _poly_eval(quad_coef, t)
            # else: leave as NaN — HGBR handles missing values natively.

        df[f"{c}_lin"]  = lin_vals
        df[f"{c}_quad"] = quad_vals

    # ── 5. Previous and next measured values (lag / lead) ────────────────────
    for c in signal_cols:
        # Sort is already (patient_id, age), so shift(-1) is the next visit.
        shifted_prev = df.groupby("patient_id")[c].shift(1)
        shifted_next = df.groupby("patient_id")[c].shift(-1)

        # Forward-fill within patient to get the last non-NaN prev value
        df[f"{c}_prev"] = (
            df.groupby("patient_id")[c]
            .transform(lambda s: s.ffill().shift(1))
        )
        df[f"{c}_next"] = (
            df.groupby("patient_id")[c]
            .transform(lambda s: s.bfill().shift(-1))
        )

    # ── 6. Patient mean LEDD ──────────────────────────────────────────────────
    df["ledd_pmean"] = df.groupby("patient_id")["ledd"].transform("mean")

    # ── 7. Per-patient ridge trajectory (off+5.9 and on_corr vs age) ─────────
    # Weighted line fitted on (off + 5.9) and on_corr together vs age,
    # slope penalised toward 2.8 points/year (prior from clinical literature).
    traj_vals  = np.full(len(df), np.nan)
    slope_vals = np.full(len(df), np.nan)

    for pat_id, grp in df.groupby("patient_id"):
        idx  = grp.index
        ages = grp["age"].values.astype(float)

        # Combine off (shifted) and on_corr into a single signal array
        y_off = grp["off"].values.astype(float) + 5.9
        y_on  = grp["on_corr"].values.astype(float)

        # Stack all available (age, value) pairs from both sources
        both_ages = np.concatenate([ages, ages])
        both_vals = np.concatenate([y_off, y_on])
        mask = np.isfinite(both_ages) & np.isfinite(both_vals)
        a_fit = both_ages[mask]
        v_fit = both_vals[mask]

        if len(a_fit) >= 2:
            intercept, slope = _ridge_weighted_line(a_fit, v_fit)
            traj_vals[idx]  = intercept + slope * ages
            slope_vals[idx] = slope
        # else: leave traj/traj_slope as NaN — HGBR handles missing natively.

    df["traj"]       = traj_vals
    df["traj_slope"] = slope_vals

    # ── 8. Upper-envelope features ────────────────────────────────────────────
    off_env_vals  = np.full(len(df), np.nan)
    on_env_vals   = np.full(len(df), np.nan)
    comb_env_vals = np.full(len(df), np.nan)

    for pat_id, grp in df.groupby("patient_id"):
        idx  = grp.index
        ages = grp["age"].values.astype(float)
        off_v     = grp["off"].values.astype(float)
        on_corr_v = grp["on_corr"].values.astype(float)

        off_env_vals[idx]  = _envelope_line(ages, off_v,     cut=2.0)
        on_env_vals[idx]   = _envelope_line(ages, on_corr_v, cut=4.0)

        # Combined: use off where available, else on_corr
        comb = np.where(np.isfinite(off_v), off_v, on_corr_v)
        comb_env_vals[idx] = _envelope_line(ages, comb, cut=3.0)

    df["off_env"]        = off_env_vals
    df["on_env"]         = on_env_vals
    df["comb_env"]       = comb_env_vals
    df["off_minus_env"]  = df["off"]  - df["off_env"]
    df["traj_minus_env"] = df["traj"] - df["off_env"]

    # ── Restore original row order ────────────────────────────────────────────
    df = df.set_index("index").loc[original_index].reset_index(drop=True)

    # ── Drop identifier / label columns not used as features ─────────────────
    drop_cols = ["patient_id", "Index", "target",
                 "index",   # from reset_index(drop=False) above
                 ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Convert string columns to category so HGBR can handle them natively ──
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].astype("category")

    return df

# ── Two-stage stacking regressor ─────────────────────────────────────────────

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold


def _s1_extra_features(
    s1: np.ndarray,
    t_first: np.ndarray,
    patient_id: np.ndarray,
) -> pd.DataFrame:
    """
    Build the 5 stage-1-derived features used by stage 2 (and predict).

    Columns: s1, s1_pmean, s1_lin, s1_quad, s1_residual.
    All computed per-patient; NaN fallback when not enough points.
    """
    df = pd.DataFrame({
        "patient_id": patient_id,
        "t_first":    t_first,
        "s1":         s1,
    })

    df["s1_pmean"] = df.groupby("patient_id")["s1"].transform("mean")

    lin_vals  = np.full(len(df), np.nan)
    quad_vals = np.full(len(df), np.nan)

    for pid, grp in df.groupby("patient_id"):
        idx   = grp.index
        t     = grp["t_first"].values.astype(float)
        v     = grp["s1"].values.astype(float)

        lin_c  = _safe_polyfit(t, v, deg=1)
        quad_c = _safe_polyfit(t, v, deg=2)

        if lin_c  is not None:
            lin_vals[idx]  = _poly_eval(lin_c,  t)
        if quad_c is not None:
            quad_vals[idx] = _poly_eval(quad_c, t)

    df["s1_lin"]      = lin_vals
    df["s1_quad"]     = quad_vals
    df["s1_residual"] = df["s1"] - df["s1_quad"]   # NaN when quad is NaN

    return df[["s1", "s1_pmean", "s1_lin", "s1_quad", "s1_residual"]]


def _make_hgbr(
    max_iter: int = 800,
    learning_rate: float = 0.03,
    min_samples_leaf: int = 40,
    l2_regularization: float = 1.0,
) -> "HistGradientBoostingRegressor":
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        random_state=0,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
    )


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage stacking regressor with per-patient quadratic smoothing.

    Expected input X: make_features(df) output PLUS two helper columns:
      - 'patient_id' : used only for GroupKFold grouping and per-patient ops.
      - 't_first'    : already present in make_features output.
    patient_id is dropped before any HGBR call.

    Parameters
    ----------
    s2_min_samples_leaf : int
        min_samples_leaf for stage-2 HGBR (stage-1 always uses 40).
    s2_l2_regularization : float
        l2_regularization for stage-2 HGBR (stage-1 always uses 1.0).
    n_inner_splits : int
        Number of GroupKFold folds for OOF stage-1 predictions.
    smooth_min_visits : int
        Minimum number of visits in a patient before applying quadratic smoothing.
    """

    def __init__(
        self,
        s2_min_samples_leaf: int = 40,
        s2_l2_regularization: float = 1.0,
        n_inner_splits: int = 5,
        smooth_min_visits: int = 3,
    ):
        self.s2_min_samples_leaf   = s2_min_samples_leaf
        self.s2_l2_regularization  = s2_l2_regularization
        self.n_inner_splits        = n_inner_splits
        self.smooth_min_visits     = smooth_min_visits

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _feat_cols(X: pd.DataFrame) -> list[str]:
        """Return the model-feature columns (drop patient_id)."""
        return [c for c in X.columns if c != "patient_id"]

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y) -> "TwoStageRegressor":
        X      = pd.DataFrame(X).copy()
        y      = np.asarray(y, dtype=float)
        pid    = X["patient_id"].values
        t_fst  = X["t_first"].values.astype(float)
        feat   = self._feat_cols(X)
        X_feat = X[feat]

        # ── Stage 1 ──────────────────────────────────────────────────────────
        self.stage1_ = _make_hgbr()

        # Out-of-fold predictions via inner GroupKFold
        oof_s1 = np.full(len(y), np.nan)
        inner_cv = GroupKFold(n_splits=self.n_inner_splits)
        for tr_idx, te_idx in inner_cv.split(X_feat, y, groups=pid):
            m = clone(self.stage1_)
            m.fit(X_feat.iloc[tr_idx], y[tr_idx])
            oof_s1[te_idx] = m.predict(X_feat.iloc[te_idx])

        # ── Stage-1-derived features ─────────────────────────────────────────
        s1_feats = _s1_extra_features(oof_s1, t_fst, pid)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        X_s2 = pd.concat(
            [X_feat.reset_index(drop=True), s1_feats.reset_index(drop=True)],
            axis=1,
        )
        self.stage2_ = _make_hgbr(
            min_samples_leaf=self.s2_min_samples_leaf,
            l2_regularization=self.s2_l2_regularization,
        )
        self.stage2_.fit(X_s2, y)

        # ── Refit stage 1 on all training rows ───────────────────────────────
        self.stage1_.fit(X_feat, y)

        # Store feature column order for predict-time alignment
        self.feat_cols_   = feat
        self.s2_cols_     = X_s2.columns.tolist()

        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X      = pd.DataFrame(X).copy()
        pid    = X["patient_id"].values
        t_fst  = X["t_first"].values.astype(float)
        X_feat = X[self.feat_cols_]

        # Stage-1 predictions
        s1_pred  = self.stage1_.predict(X_feat)

        # Stage-1-derived features
        s1_feats = _s1_extra_features(s1_pred, t_fst, pid)
        X_s2     = pd.concat(
            [X_feat.reset_index(drop=True), s1_feats.reset_index(drop=True)],
            axis=1,
        )[self.s2_cols_]

        raw_pred = self.stage2_.predict(X_s2)

        # ── Per-patient quadratic smoothing ───────────────────────────────────
        smoothed = raw_pred.copy()
        df_sm    = pd.DataFrame({
            "patient_id": pid,
            "t_first":    t_fst,
            "raw":        raw_pred,
        })
        for p_id, grp in df_sm.groupby("patient_id"):
            if len(grp) < self.smooth_min_visits:
                continue
            idx  = grp.index
            t    = grp["t_first"].values.astype(float)
            v    = grp["raw"].values.astype(float)
            coef = _safe_polyfit(t, v, deg=2)
            if coef is not None:
                smoothed[idx] = _poly_eval(coef, t)

        return np.clip(smoothed, 0.0, 132.0)
