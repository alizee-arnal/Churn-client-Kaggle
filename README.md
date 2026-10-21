# Prédiction du churn client (Telco)

Analyse et modélisation du départ des clients d'un opérateur télécom, à partir du dataset Kaggle [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7 043 clients, 21 variables).

**Objectif :** identifier les clients susceptibles de résilier leur abonnement (churn) et comprendre les facteurs qui expliquent leur départ, afin de cibler des actions de rétention.

## Résultats en bref

- **26,5 %** des clients sont partis : le jeu de données est déséquilibré.
- Le modèle retenu est la **régression logistique** : son AUC sur le test (0,839) est proche de celle de la Random Forest (0,845), et elle reste interprétable.
- Au seuil de 0,5, elle détecte **78 % des clients qui partent** (rappel), avec une précision de 51 %.
- Facteurs qui augmentent le risque de départ : fibre optique, paiement par chèque électronique, facturation dématérialisée, options streaming.
- Facteurs qui le réduisent : ancienneté, contrat d'un ou deux ans, options sécurité en ligne et support technique.

### Comparaison des modèles

Validation croisée stratifiée à 5 folds sur le jeu d'entraînement, puis évaluation sur le jeu de test (20 % des données).

| Modèle | AUC (validation croisée) | Rappel (CV) | Précision (CV) | AUC (test) |
|---|---|---|---|---|
| Baseline (dummy) | 0,507 | 0,278 | 0,275 | 0,516 |
| Régression logistique | 0,845 | 0,794 | 0,521 | 0,839 |
| Random Forest | 0,846 | 0,759 | 0,542 | 0,845 |
| Gradient Boosting | 0,846 | 0,796 | 0,526 | 0,839 |

Les trois modèles ont des performances très proches. Le plus simple est donc retenu pour pouvoir lire les coefficients.

### Choix du seuil de décision

Le seuil dépend du coût d'une action de rétention par rapport à celui d'un client perdu.

| Seuil | Clients alertés | Rappel | Précision |
|---|---|---|---|
| 0,3 | 805 | 93 % | 43 % |
| 0,4 | 687 | 86 % | 47 % |
| **0,5** | 571 | 78 % | 51 % |
| 0,6 | 476 | 70 % | 55 % |
| 0,7 | 362 | 60 % | 62 % |

### Facteurs de risque (odds ratios de la régression logistique)

| Variable | Odds ratio |
|---|---|
| Internet fibre optique | 3,06 |
| Paiement par chèque électronique | 1,49 |
| Streaming films / TV | 1,48 / 1,45 |
| Facturation sans papier | 1,40 |
| Ancienneté (tenure) | 0,47 |
| Contrat 1 an / 2 ans | 0,49 / 0,25 |

Un odds ratio supérieur à 1 augmente la « cote » de départ, un odds ratio inférieur à 1 la réduit.

## Méthode

1. **Préparation** : suppression de `customerID` ; conversion de `TotalCharges` en numérique ; fusion de « No internet service » et « No phone service » avec « No » ; suppression de `TotalCharges`, redondante avec `tenure × MonthlyCharges`.
2. **Exploration** : répartition du churn, taux de churn par variable, distributions, corrélations (figures 01 à 07).
3. **Modélisation** : pipeline scikit-learn (standardisation des variables numériques, one-hot encoding des catégorielles), séparation train/test stratifiée, `class_weight="balanced"` pour compenser le déséquilibre.
4. **Évaluation** : AUC, rappel, précision, F1, matrice de confusion, courbe précision/rappel selon le seuil (figures 08 à 11).

## Structure du projet

```
.
├── download_kaggle_data.py   # télécharge le dataset dans data/raw/
├── exploration.py            # analyse exploratoire -> figures 01 à 07
├── modelisation.py           # comparaison des modèles -> figures 08 à 11
├── figures/                  # graphiques générés
└── data/raw/                 # données (non versionnées)
```

## Installation et utilisation

Prérequis : Python 3.10+.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kaggle
```

1. **Données** : crée un token sur [kaggle.com/settings](https://www.kaggle.com/settings), place `kaggle.json` dans `~/.kaggle/`, puis lance :
   ```bash
   python download_kaggle_data.py
   ```
2. **Exploration** :
   ```bash
   python exploration.py
   ```
3. **Modélisation** :
   ```bash
   python modelisation.py
   ```

Les scripts se lancent depuis la racine du projet. Les figures sont enregistrées dans `figures/`.

## Pistes d'amélioration

- Optimiser les hyperparamètres (grid search) et calibrer les probabilités.
- Chiffrer le seuil optimal avec le coût d'une offre de rétention et la valeur d'un client.
- Tester d'autres modèles (XGBoost, LightGBM) et des interactions entre variables.
