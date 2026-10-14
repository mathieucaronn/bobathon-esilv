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
