# Notre rendu — équipe M&C

**Notre code, ce sont les notebooks du dossier [`notebooks_finaux/`](notebooks_finaux/).** Ils sont déjà exécutés : les résultats s'affichent sans rien relancer. Chaque section explique ce que l'on fait, puis montre le code et le résultat ; les réponses aux questions du workbook sont à la fin de chaque notebook.

| Ordre | Notebook | Contenu | Meilleur RMSE |
|---|---|---|---|
| 1 | [`01_day1.ipynb`](notebooks_finaux/01_day1.ipynb) | Exploration des données (EDA) et modèle dummy | 16,48 |
| 2 | [`02_day2.ipynb`](notebooks_finaux/02_day2.ipynb) | Étapes 1 à 7 du Day 2 : Ridge, validation par patient, HistGradientBoosting, skrub, soumission | 7,43 |
| 3 | [`03_ameliorations.ipynb`](notebooks_finaux/03_ameliorations.ipynb) | Après le guide : trajectoire de chaque patient, empilement, enveloppe supérieure | **3,37** |

## Résultat

| Modèle | RMSE validation par patient (GroupKFold) | Kaggle public | Kaggle privé |
|---|---|---|---|
| `07_tabular` (fin du guide) | 7,43 | | |
| **`10_env_stack` (notre soumission finale)** | **3,37** | **3,339** | **3,192** |

Rapport Skore Hub du modèle final : https://skore.probabl.ai/M&C/bobathon-esilv/estimators/88595 (workspace `M&C`, projet `bobathon-esilv`).

## Ce que nous avons fait

1. **Suivi du guide (Day 1 et Day 2)** : dummy (16,5) → Ridge (10,4) → HistGradientBoosting qui garde les valeurs manquantes (7,4) → skrub `tabular_pipeline` et DataOps.
2. **Validation honnête** : `GroupKFold` sur `patient_id`, car Kaggle teste sur des patients nouveaux. Elle prédit le score Kaggle à 0,03 près, alors que le découpage aléatoire est trop optimiste.
3. **L'idée clé** : pour un patient, le vrai score OFF suit presque parfaitement une courbe lisse de l'âge (erreur de 0,33). Nous reconstruisons cette trajectoire à partir des mesures `on`/`off` de **toutes ses visites**, en corrigeant leurs biais (`off` sous-estime le vrai score, `on` dépend du délai depuis la prise du médicament).
4. **Le modèle final** : variables par patient + empilement en deux étages (prédictions de l'étage 1 hors échantillon) + lissage de la trajectoire de chaque patient.
5. **Pas de fuite** : aucune variable n'utilise `target`, et `patient_id` / `Index` ne sont jamais des variables du modèle.

## Organisation du dossier

| Élément | Rôle |
|---|---|
| [`notebooks_finaux/`](notebooks_finaux/) | **Notre code, à lire en premier** |
| [`features.py`](features.py) | Construction des variables par patient et modèle `TwoStageRegressor` (importé par le notebook 3) |
| [`cv_splits.py`](cv_splits.py) | Chargement des données et découpage GroupKFold commun à tous les modèles |
| [`submissions/`](submissions/) | Fichiers envoyés sur Kaggle (`10_env_stack.csv` = soumission finale) |
| [`journal/JOURNAL.md`](journal/JOURNAL.md) | Journal des expériences tenu par IBM Bob (scores et liens Skore Hub) |
| [`archives_py/`](archives_py/) | Scripts `.py` d'origine, remplacés par les notebooks (sauvegarde) |
| `docs/`, `tests/`, `scripts/` | Consignes et outils fournis par le lab |

**Pour réexécuter les notebooks** : installer l'environnement comme dans [docs/SETUP.md](docs/SETUP.md), placer les CSV de Kaggle dans `data/`, puis exécuter les notebooks dans l'ordre. Les envois vers Skore Hub sont désactivés (`PUSH = False`) car les rapports y sont déjà.

---

# Consignes du lab (README d'origine)

<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="60">
</p>
# Help science build better Parkinson's evaluation

*Predicting and correcting bias in motor score evaluation — for each visit, predict the unbiased (“true”) OFF MDS-UPDRS motor score.*

## Getting started

```bash
git clone https://github.com/probabl-ai/bobathon-esilv.git
cd bobathon-esilv
```

The lab has three steps. Do them **in order**: each one assumes the previous one is finished.


| Step | Guide                     | What you do                                                                                                |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1    | [SETUP.md](docs/SETUP.md) | Join Kaggle, form a team, install Python, create the shared Skore Hub workspace, install skore and sign in |
| 2    | [DAY1.md](docs/DAY1.md)   | Download the data, explore it (EDA), push a dummy baseline to Skore Hub                                    |
| 3    | [DAY2.md](docs/DAY2.md)   | Build real models, evaluate them with patient-grouped cross-validation, submit to Kaggle                   |


### Step 1 — Setup ([SETUP.md](docs/SETUP.md))

Follow every section of [SETUP.md](docs/SETUP.md):

1. **Join Kaggle and form a team** (maximum four people). Remember the exact team name.
2. **Install Python 3.12 or newer** and check it with `python --version` (or `python3 --version`).
3. **Create one Skore Hub workspace per team** on [skore.probabl.ai](https://skore.probabl.ai), named like the Kaggle team. One teammate creates it and invites the others.
4. **Allow Bob to use the lab skills** (tick **Skill** in Bob's Permissions menu), then **install skore and sign in** from the repo root.

#### Check your setup before moving on

Run the setup tests, with the virtual environment still active:

```bash
python -m pytest
```

See [SETUP.md step 5](docs/SETUP.md#5-install-skore-and-check-your-setup) for details.

**You cannot proceed to Day 1 until the last line says** `all passed`**.** Without a working `.skore`, you cannot push reports to Skore Hub, and every Kaggle Submission needs a Hub report URL.

### Step 2 — Day 1 ([DAY1.md](docs/DAY1.md))

1. Download the CSVs from the [competition Data tab](https://www.kaggle.com/competitions/bobathon-esilv/data) into `data/`.
2. Explore the data in Bob (EDA) and go through the workbook questions.
3. Build a **dummy mean** baseline, evaluate it with `skore.evaluate`, and `put` it to Skore Hub as `01_dummy`. Every later model must beat this score.

### Step 3 — Day 2 ([DAY2.md](docs/DAY2.md))

1. Train a **Ridge** linear model, compare it with the dummy, and make your **first Kaggle Submission**.
2. Switch to **patient-grouped cross-validation** (`GroupKFold`) so the score matches the Kaggle split.
3. Try **HistGradientBoosting** (missing values as signal), skrub `**tabular_pipeline`** (categorical columns), and skrub **DataOps** (groups baked into the graph).
4. Fit your best model on all training visits and **submit again**.

## Documents

- [SETUP.md](docs/SETUP.md) - Kaggle team, Python, Skore Hub workspace, skore install and sign-in
- [DAY1.md](docs/DAY1.md) - data, exploratory data analysis (EDA), dummy baseline
- [DAY2.md](docs/DAY2.md) - models, grouped cross-validation, Kaggle Submissions
- [CONTEXT.md](docs/CONTEXT.md) - scientific background, goals, and modelling pitfalls for the Parkinson's dataset
- Kaggle competition: [bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)

## Submission rules

All code must be written and run in **Bob** (IDE or CLI), with evaluation through **skore**. Every Kaggle upload must put a Skore Hub EstimatorReport URL (`https://skore.probabl.ai/…`) in the Submission Description, and each Submission needs a **new** report key (`01_dummy`, `02_ridge`, …). Each experiment writes its file to `submissions/<report-key>.csv` (e.g. `submissions/02_ridge.csv`), so the filename matches the report key and a later model never overwrites an earlier submission. See the [competition rules](https://www.kaggle.com/competitions/bobathon-esilv/rules) for eligibility and what makes a Submission valid.
