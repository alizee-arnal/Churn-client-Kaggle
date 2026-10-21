from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
plt.style.use('ggplot')

DATA_PATH = Path("data") / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_DIR = Path("figures")
RANDOM_STATE = 42

COLOR_STAY = "#3b6fb6"
COLOR_CHURN = "#c0504d"

pd.set_option("display.width", None)

#import and preparation (same steps as the exploration)

df = pd.read_csv(DATA_PATH)
df.drop(columns=["customerID"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

#"No internet service" and "No phone service" just repeat InternetService / PhoneService: map them to "No"
df.replace({"No internet service": "No", "No phone service": "No"}, inplace=True)

#TotalCharges ≈ tenure x MonthlyCharges: redundant variable that blurs the logistic regression coefficients
df.drop(columns=["TotalCharges"], inplace=True)

y = (df.pop("Churn") == "Yes").astype(int)
X = df

numeric_cols = ["tenure", "MonthlyCharges"]
categorical_cols = [col for col in X.columns if col not in numeric_cols]

#1. train / test split (stratify = keep ~27% churn in both)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
)
print(f"Train : {len(X_train)} clients | Test : {len(X_test)} clients | churn train {y_train.mean():.1%} / test {y_test.mean():.1%}")

#2. feature preprocessing in a pipeline (fitted on the train set only)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
])

models = {
    "Baseline (dummy)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    "Régression logistique": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_depth=4, class_weight="balanced", random_state=RANDOM_STATE,
    ),
}
pipelines = {name: Pipeline([("prep", preprocessor), ("model", model)]) for name, model in models.items()}

#3. model comparison with cross-validation (on the train set only)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["roc_auc", "recall", "precision", "f1", "accuracy"]

cv_results = {}
for name, pipe in pipelines.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_results[name] = {metric: scores[f"test_{metric}"].mean() for metric in scoring}
    cv_results[name]["roc_auc_std"] = scores["test_roc_auc"].std()

cv_table = pd.DataFrame(cv_results).T.round(3).sort_values("roc_auc", ascending=False)
print("\nValidation croisée (moyenne sur 5 folds) :")
print(cv_table)

#4. final training and evaluation on the test set

for pipe in pipelines.values():
    pipe.fit(X_train, y_train)

FIGURES_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
ax.set_title("Courbes ROC sur le jeu de test")
ax.set_xlabel("Taux de faux positifs (fidèles signalés à tort)")
ax.set_ylabel("Taux de vrais positifs (partants détectés)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "08_courbes_roc.png", dpi=150)

#chosen model: logistic regression (performance close to boosting, but interpretable)

CHOSEN = "Régression logistique"
chosen_pipe = pipelines[CHOSEN]
proba_test = chosen_pipe.predict_proba(X_test)[:, 1]

print("\nAUC sur le test :")
for name, pipe in pipelines.items():
    print(f"  {name:<22} {roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]):.3f}")

print(f"\nRapport de classification - {CHOSEN} (seuil 0.5) :")
print(classification_report(y_test, chosen_pipe.predict(X_test), target_names=["Restent", "Partent"]))

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, chosen_pipe.predict(X_test), display_labels=["Restent", "Partent"], cmap="Blues", colorbar=False, ax=ax,
)
ax.grid(False)
ax.set_title(f"Matrice de confusion - {CHOSEN}")
ax.set_xlabel("Prédiction")
ax.set_ylabel("Réalité")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "09_matrice_confusion.png", dpi=150)

#5. threshold choice: precision / recall trade-off

precision, recall, thresholds = precision_recall_curve(y_test, proba_test)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(thresholds, precision[:-1], color=COLOR_STAY, linewidth=2, label="Précision (alertes justes)")
ax.plot(thresholds, recall[:-1], color=COLOR_CHURN, linewidth=2, label="Rappel (partants détectés)")
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
ax.set_title(f"Effet du seuil de décision - {CHOSEN}")
ax.set_xlabel("Seuil de probabilité au-dessus duquel on prédit « part »")
ax.set_ylabel("Score")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "10_seuil_precision_rappel.png", dpi=150)

print("Selon le seuil choisi :")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (proba_test >= threshold).astype(int)
    tp = ((pred == 1) & (y_test == 1)).sum()
    print(
        f"  seuil {threshold:.1f} : {pred.sum():4d} clients alertés, "
        f"{tp} vrais partants sur {y_test.sum()} (rappel {tp / y_test.sum():.0%}, précision {tp / max(pred.sum(), 1):.0%})"
    )

#6. interpretation: logistic regression coefficients

feature_names = chosen_pipe.named_steps["prep"].get_feature_names_out()
feature_names = [name.split("__", 1)[1] for name in feature_names]
coefs = pd.Series(chosen_pipe.named_steps["model"].coef_[0], index=feature_names).sort_values()

print("\nOdds ratios (x combien la variable multiplie la « cote » de départ) :")
print(np.exp(coefs).sort_values(ascending=False).round(2))

fig, ax = plt.subplots(figsize=(8, 8))
colors = [COLOR_STAY if value < 0 else COLOR_CHURN for value in coefs]
ax.barh(coefs.index, coefs.values, color=colors)
ax.axvline(0, color="grey", linewidth=1)
ax.set_title("Coefficients de la régression logistique\n(rouge = augmente le risque, bleu = le réduit)")
ax.set_xlabel("Coefficient (log-odds)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "11_coefficients_logistique.png", dpi=150)

print(f"\nFigures enregistrées dans {FIGURES_DIR.resolve()}")
plt.show()
