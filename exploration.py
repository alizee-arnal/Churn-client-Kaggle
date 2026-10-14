from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')

#importation données


DATA_PATH = Path("data") / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_DIR = Path("figures")

COLOR_STAY = "#3b6fb6"
COLOR_CHURN = "#c0504d"
CHURN_LABELS = {"No": "Restent", "Yes": "Partent"}
CHURN_PALETTE = {"Restent": COLOR_STAY, "Partent": COLOR_CHURN}

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv(DATA_PATH)

##print(df.head(10))


#préparation data

df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

FIGURES_DIR.mkdir(exist_ok=True)
is_churn = (df["Churn"] == "Yes").astype(int)
global_churn_rate = is_churn.mean() * 100

#version lisible des données pour les graphiques (df reste intact pour l'encodage)
df_plot = df.assign(
    Statut=df["Churn"].map(CHURN_LABELS),
    SeniorCitizen=df["SeniorCitizen"].map({0: "No", 1: "Yes"}),
)

#1. répartition du churn

churn_counts = df["Churn"].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Restent", "Partent"], churn_counts[["No", "Yes"]], color=[COLOR_STAY, COLOR_CHURN])
for i, count in enumerate(churn_counts[["No", "Yes"]]):
    ax.text(i, count, f"{count} ({count / len(df):.0%})", ha="center", va="bottom")
ax.set_title("Répartition des clients")
ax.set_ylabel("Nombre de clients")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_repartition_churn.png", dpi=150)

#2. taux de churn par variable catégorielle

categorical_cols = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
    "Partner",
    "Dependents",
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, col in zip(axes.flat, categorical_cols):
    churn_rate = is_churn.groupby(df_plot[col]).mean().mul(100).sort_values()
    ax.barh(churn_rate.index.astype(str), churn_rate.values, color=COLOR_CHURN)
    ax.axvline(global_churn_rate, color="grey", linestyle="--", linewidth=1)
    for y, rate in enumerate(churn_rate.values):
        ax.text(rate + 0.5, y, f"{rate:.0f}%", va="center")
    ax.set_title(col)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Taux de churn (%)")
fig.suptitle(f"Taux de churn par catégorie (pointillés = moyenne {global_churn_rate:.0f}%)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_taux_churn_par_categorie.png", dpi=150)

#3. distribution des variables numériques selon le churn

numeric_cols = {
    "tenure": "Ancienneté (mois)",
    "MonthlyCharges": "Montant mensuel ($)",
    "TotalCharges": "Montant total ($)",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (col, label) in zip(axes, numeric_cols.items()):
    sns.kdeplot(
        data=df_plot, x=col, hue="Statut", hue_order=["Restent", "Partent"], palette=CHURN_PALETTE,
        common_norm=False, fill=True, alpha=0.3, linewidth=2, ax=ax,
    )
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Densité")
fig.suptitle("Distribution selon le churn")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_distributions_numeriques.png", dpi=150)

#4. taux de churn selon l'ancienneté

tenure_bins = pd.cut(
    df["tenure"],
    bins=[-1, 6, 12, 24, 36, 48, 60, 72],
    labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
)
churn_by_tenure = is_churn.groupby(tenure_bins, observed=True).mean().mul(100)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(churn_by_tenure.index.astype(str), churn_by_tenure.values, color=COLOR_CHURN, linewidth=2, marker="o", markersize=8)
for x, rate in enumerate(churn_by_tenure.values):
    ax.text(x, rate + 1.5, f"{rate:.0f}%", ha="center")
ax.set_title("Taux de churn selon l'ancienneté")
ax.set_xlabel("Ancienneté (mois)")
ax.set_ylabel("Taux de churn (%)")
ax.set_ylim(0, 60)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "04_churn_par_anciennete.png", dpi=150)

#5. contrat x montant mensuel

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(
    data=df_plot, x="Contract", y="MonthlyCharges", hue="Statut", hue_order=["Restent", "Partent"],
    palette=CHURN_PALETTE, order=["Month-to-month", "One year", "Two year"], ax=ax,
)
ax.set_title("Montant mensuel par type de contrat et churn")
ax.set_xlabel("Type de contrat")
ax.set_ylabel("Montant mensuel ($)")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Churn")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_contrat_montant.png", dpi=150)

#encodage des variables texte en 0/1

df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

#6. corrélations

df_corr = df_encoded.corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(df_corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
ax.set_title("Corrélations entre toutes les variables")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_heatmap_correlations.png", dpi=150)

#7. corrélation de chaque variable avec le churn

churn_corr = df_corr["Churn_Yes"].drop("Churn_Yes").sort_values()
print("\nCorrélation avec le churn :")
print(churn_corr)

fig, ax = plt.subplots(figsize=(8, 9))
colors = [COLOR_STAY if value < 0 else COLOR_CHURN for value in churn_corr]
ax.barh(churn_corr.index, churn_corr.values, color=colors)
ax.axvline(0, color="grey", linewidth=1)
ax.set_title("Corrélation avec le churn (rouge = favorise le départ)")
ax.set_xlabel("Coefficient de corrélation")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "07_correlation_churn.png", dpi=150)

print(f"\nFigures enregistrées dans {FIGURES_DIR.resolve()}")
plt.show()
