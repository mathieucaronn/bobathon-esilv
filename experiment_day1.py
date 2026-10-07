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
