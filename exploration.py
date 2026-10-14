import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColonnes:", df.columns.tolist())
print("\nAperçu:")
print(df.head())
print("\nInfos:")
print(df.info())
print("\nValeurs manquantes:")
print(df.isnull().sum())
print("\nTaux de churn:")
print(df["Churn"].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
plt.title("Distribution du Churn")
plt.xlabel("Churn")
plt.ylabel("Nombre de clients")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.show()
