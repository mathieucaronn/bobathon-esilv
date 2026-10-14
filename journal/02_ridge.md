# 02_ridge — Simple linear model (Ridge) vs dummy

## Question / hypothesis

Does a median-imputed Ridge regression on numeric clinical features beat the dummy mean baseline on the random row holdout?

## Motivation

Day 1 established the floor: `DummyRegressor(mean)` predicts the same value for every visit. The lab guide (Day 2, step 1) asks us to check whether the numeric clinical variables (`age`, `on`, `off`, `ledd`, timing delays) carry an additive linear signal. Ridge is the right first check: it fits a weighted sum of the columns, with an `alpha` penalty that prevents one noisy column from dominating.

The comparison report (both estimators in one `evaluate` call) shows directly whether the linear signal is real before we invest in more complex models.

## Method

- Features: same eight numeric columns as Day 1 (`sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`).
- Preprocessing: `SimpleImputer(strategy="median")` to fill NaNs — required because Ridge cannot handle missing values.
- Estimator: `Ridge(alpha=1.0)`.
- Pipeline: `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))`.
- Evaluation: `skore.evaluate({"dummy": dummy, "ridge": ridge}, X, y, splitter=0.2)` — random 20 % row holdout (same setup as Day 1 dummy, deliberately, so scores are directly comparable).
- Hub key: `02_ridge` pushed via `project.put("02_ridge", report)`.
- Submission: Ridge refitted on all training rows, predicts `X_test`, written to `submissions/02_ridge.csv`.

Note: `splitter=0.2` can place visits from the same patient on both sides (leakage). That is intentional here — it is the same split as the dummy, so the comparison is apples-to-apples. Patient-grouped CV is the next step (B1).

## Risks

- If Ridge does not beat dummy here, it may be because the linear signal is weak OR because the random split is too easy (leakage inflating both scores equally). Grouped CV (B1) will clarify.
- Median imputation hides the information in missingness. This is fine for the linear baseline; `HistGradientBoostingRegressor` (B2) will test keeping the holes.
- Features are not scaled, so raw Ridge coefficients cannot be compared across columns with different units.

## Status

- **State:** done
- **Approved by user on:** 2025-07-14
- **Headline result:** RMSE ridge 10.387 vs RMSE dummy 16.478 — **+37 % improvement** on random 20 % row holdout (splitter=0.2)
- **Implication for next iteration:** Ridge beats dummy convincingly, confirming a real linear signal in the numeric features. However the random row holdout leaks patients across train/test sides. The true gain may be smaller; patient-grouped CV (B1) will give the honest estimate.
