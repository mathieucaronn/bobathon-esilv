"""
Jour 1 — Baseline dummy
Génère submissions/01_dummy.csv et pousse le rapport sur Skore Hub.
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.base import clone
from skore import evaluate, Project, login

# ── 1. Credentials ──────────────────────────────────────────────────────────
cfg = json.loads(Path(".skore").read_text(encoding="utf-8"))
os.environ["SKORE_HUB_URI"]     = cfg["hub_url"]
os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]

# ── 2. Données ───────────────────────────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test  = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[feature_cols]
y = visits["target"]

# ── 3. Dummy baseline ────────────────────────────────────────────────────────
dummy  = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)

rmse = report.metrics.rmse()
print(f"RMSE dummy  : {rmse:.3f}")
print(f"Std target  : {y.std():.3f}  (référence : tout modèle réel doit faire mieux)")

# ── 4. Push Skore Hub ────────────────────────────────────────────────────────
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
print("Rapport pushé → consulte l'URL imprimée ci-dessus")

# ── 5. Fichier de soumission Kaggle ──────────────────────────────────────────
final = clone(dummy).fit(X, y)

submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])

Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/01_dummy.csv", index=False)
print("Fichier généré : submissions/01_dummy.csv")
print(f"Shape : {submission.shape}  — colonnes : {list(submission.columns)}")

# ── Réponses workbook — Day 1, section 2 (EDA) ──────────────────────────────
# Q1. Train : 44 590 visites, 5 576 patients. Test : 11 013 visites, 1 395 patients.
#     Environ 7 visites par patient en médiane, 12 au maximum.
# Q2. Colonnes les plus vides : time_since_intake_off (79 %), time_since_intake_on (46 %),
#     off (42 %). 30 % des visites n'ont pas de score on. Aucune visite n'a ni on ni off.
# Q3. target va de 0 à 109,5 (moyenne 37,5, écart-type 16,5). Jamais près de 132.
#     Distribution presque symétrique (skewness ≈ 0,05).
# Q4. off est le plus corrélé à target (0,89), puis on (0,69) et la durée de la maladie
#     age - age_at_diagnosis (0,54). off n'est pas une copie : il s'écarte de target
#     d'environ 10 points (RMSE). Les délais de prise sont peu corrélés seuls (≈ 0).
# Q5. ledd manque plus dans la cohorte B (43 %) que dans la A (36 %) : ce n'est pas
#     au hasard, ça dépend du protocole de chaque cohorte.
# Q6. On ne donne pas Index ni patient_id (identifiants : le modèle les mémoriserait)
#     ni target (c'est la réponse). rater_id est à utiliser avec prudence.

# ── Réponses workbook — Day 1, section 3 (dummy) ────────────────────────────
# Q1. RMSE dummy = 16,48, presque égal à l'écart-type de target (16,50).
#     Normal : prédire toujours la moyenne donne une erreur égale à l'écart-type.
# Q2. Le graphe prédit vs réel est une ligne horizontale : même prédiction pour tous.
#     Un modèle parfait aurait tous ses points sur la diagonale y = x.
# Q3. Score à battre : 16,48. Un gain n'est réel que s'il dépasse la variation entre
#     folds (≈ 0,3 pour le dummy en GroupKFold) : en dessous, c'est du bruit.
