# JOURNAL

## Status

- **Project / dataset:** Parkinson's disease (PD) multi-cohort visits — predict true OFF MDS-UPDRS motor score
- **Goal:** Minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 03_ridge_gcv — done
- **Last result:** Ridge grouped CV RMSE 10.437 ± 0.197 vs dummy 16.499 ± 0.343 (+37 % on GroupKFold)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-14
  - env manager: pip+venv - recorded: 2025-07-14
  - agent feature: installed - recorded: 2025-07-14
  - optional features: none - recorded: 2025-07-14
  - package name (`src/<pkg>/`): n/a (flat scripts) - recorded: 2025-07-14
  - skore mode: hub - recorded: 2025-07-14
  - skore hub workspace: M&C - recorded: 2025-07-14
  - skore mlflow tracking uri: n/a - recorded: 2025-07-14
  - student prior: some-sklearn - recorded: 2025-07-14
  - CV splitter family: GroupKFold(n_splits=5) on patient_id - recorded: 2025-07-14

## Data understanding (EDA)

- **Status:** done (Day 1)
- **Summary:** ~N visits across multiple patients; features include numeric clinical vars (age, on, off, ledd, timing delays), sex, and categorical vars (gene, cohort). Target is true OFF MDS-UPDRS. OFF exams are frequently missing. Holdout is by patient_id (not by visit row).
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) baseline — floor every model must beat | done | see experiment_day1.py | n/a |
| `02_ridge` | SimpleImputer + Ridge vs dummy, row holdout splitter=0.2 | done | RMSE 10.387 vs dummy 16.478 (+37 %) · hub: [02_ridge](https://skore.probabl.ai/M&C/bobathon-esilv/estimators/72541) | [design note](02_ridge.md) |
| `02_ridge_alpha0.1` | Alpha sweep — best alpha=0.1, row holdout splitter=0.2 | done | RMSE 10.387 (+37 % vs dummy) · hub: [02_ridge_alpha0.1](https://skore.probabl.ai/M&C/bobathon-esilv/estimators/72554) · CSV: submissions/02_ridge_alpha0.1.csv | n/a |
| `03_dummy_gcv` | DummyRegressor with GroupKFold(5) on patient_id | done | RMSE 16.499 ± 0.343 · hub: [03_dummy_gcv](https://skore.probabl.ai/M&C/bobathon-esilv/cross-validations/72591) | n/a |
| `03_ridge_gcv` | Ridge(alpha=0.1) + median imputer with GroupKFold(5) on patient_id | done | RMSE 10.437 ± 0.197 (+37 % vs dummy) · hub: [03_ridge_gcv](https://skore.probabl.ai/M&C/bobathon-esilv/cross-validations/72597) | n/a |

## Backlog

| # | Item | Source |
|---|---|---|
| ~~B1~~ | ~~Patient-grouped CV (GroupKFold) to block patient-level leakage~~ | resolved in 03_ridge_gcv |
| B2 | HistGradientBoostingRegressor — keep NaN as signal, no imputation | lab guide step 4 |
| B3 | tabular_pipeline (skrub) — include gene/cohort categoricals | lab guide step 5 |
