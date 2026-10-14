# JOURNAL

## Status

- **Project / dataset:** Parkinson's disease (PD) multi-cohort visits — predict true OFF MDS-UPDRS motor score
- **Goal:** Minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 08_patient_feats_gcv — done
- **Last result:** Patient-trajectory features + tuned HGBR, grouped CV RMSE 3.749 ± 0.028 (+49.5 % vs step 5)

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
| `04_hgbr_gcv` | HistGradientBoostingRegressor, NaN as signal, GroupKFold(5) | done | RMSE 7.434 ± 0.125 — beats Ridge on every fold; train gap +0.527 (mild overfit) · hub: [04_hgbr_gcv](https://skore.probabl.ai/M&C/bobathon-esilv/cross-validations/82319) | n/a |
| `05_tabular_gcv` | skrub tabular_pipeline (TableVectorizer + HGBR) with cohort & gene, GroupKFold(5) | done | RMSE 7.430 ± 0.122 — delta vs HGBR: +0.004 (within fold noise); train gap +0.580 · hub: [05_tabular_gcv](https://skore.probabl.ai/M&C/bobathon-esilv/cross-validations/83137) | n/a |
| `06_dataops_gcv` | skrub DataOps graph (CV baked in); X_op includes Index+patient_id | done | RMSE 7.573 ± 0.147 — +0.143 vs step 5 due to identity columns; train gap +1.227 (higher overfit) · hub: [06_dataops_gcv](https://skore.probabl.ai/M&C/bobathon-esilv/cross-validations/83874) | n/a |
| `07_tabular` | tabular_pipeline final fit on all training data; row holdout report + Kaggle submission | done | Test RMSE 7.295, train 6.861 (gap +0.434) · hub: [07_tabular](https://skore.probabl.ai/M&C/bobathon-esilv/estimators/84189) · CSV: submissions/07_tabular.csv | n/a |
| `08_patient_feats_gcv` | Patient-trajectory features (lag, polyfit, traj, on_corr) + tuned HGBR, GroupKFold(5) | done | RMSE 3.740 ± 0.030 after NaN-fallback fix (+49.7 % vs step 5); train gap +1.282; top feat: traj · hub: [08_patient_feats_gcv](https://skore.probabl.ai/M&C/bobathon-esilv/cross-validations/85885) | n/a |
| `08_patient_feats` | Same model, EstimatorReport (splitter=0.2) + Kaggle submission | done | Test RMSE 3.017, train 2.536 (gap +0.480) · hub: [08_patient_feats](https://skore.probabl.ai/M&C/bobathon-esilv/estimators/86353) · CSV: submissions/08_patient_feats.csv | n/a |

## Backlog

| # | Item | Source |
|---|---|---|
| ~~B1~~ | ~~Patient-grouped CV (GroupKFold) to block patient-level leakage~~ | resolved in 03_ridge_gcv |
| ~~B2~~ | ~~HistGradientBoostingRegressor — keep NaN as signal, no imputation~~ | resolved in 04_hgbr_gcv |
| ~~B3~~ | ~~tabular_pipeline (skrub) — include gene/cohort categoricals~~ | resolved in 05_tabular_gcv |
| ~~B4~~ | ~~Produce Kaggle submission from 08_patient_feats_gcv (refit on all training data)~~ | resolved in 08_patient_feats |
| B5 | Reduce overfitting gap (+1.282): try max_iter tuning or early stopping | my-pick:08_patient_feats_gcv |
| B6 | Lag feature: previous visit's target (would require careful leak-free construction) | user |
