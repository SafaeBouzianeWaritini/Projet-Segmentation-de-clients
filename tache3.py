"""
Author: Wiame Boujamaai
Role: Intelligent Cluster Analysis and Validation
Goals: Customer Segmentation for Marketing Strategy
Objectives:
Business interpretation of clusters
Mathematical validation using the Silhouette Score
Comparative analysis and critique
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
print("="*60)
print("CHARGEMENT DES DONNÉES")
print("="*60)

df = pd.read_csv("Mall_Customers_With_Clusters.csv")

# récupérer valeurs originales pour interprétation business
df_original = pd.read_csv("Mall_Customers.csv")
df_original['Cluster'] = df['Cluster']

print(f"Nombre total de clients : {len(df)}")
print(f"Nombre de clusters : {df['Cluster'].nunique()}")

print("\nRépartition des clients par cluster :")
print(df_original['Cluster'].value_counts().sort_index())


# ============================================================
# 2. INTERPRÉTATION BUSINESS DES CLUSTERS
# ============================================================
print("\n" + "="*60)
print("1. INTERPRÉTATION BUSINESS DES CLUSTERS")
print("="*60)

cluster_stats = df_original.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Gender': lambda x: (x == 'Male').sum() / len(x) * 100
}).round(2)

cluster_stats.columns = [
    'Age_mean', 'Age_std',
    'Income_mean', 'Income_std',
    'Spending_mean', 'Spending_std',
    'Male_percent'
]

print("\nStatistiques descriptives par cluster :")
print(cluster_stats)


print("\n" + "-"*40)
print("INTERPRÉTATION DES SEGMENTS CLIENT")
print("-"*40)

for cluster_id in sorted(df_original['Cluster'].unique()):
    cluster_data = df_original[df_original['Cluster'] == cluster_id]

    age_mean = cluster_data['Age'].mean()
    income_mean = cluster_data['Annual Income (k$)'].mean()
    spending_mean = cluster_data['Spending Score (1-100)'].mean()
    male_percent = (cluster_data['Gender'] == 'Male').mean() * 100

    age_cat = "jeunes" if age_mean < 35 else "âgés" if age_mean > 50 else "adultes"
    income_cat = "faible" if income_mean < 40 else "élevé" if income_mean > 70 else "moyen"
    spending_cat = "faible" if spending_mean < 40 else "élevé" if spending_mean > 60 else "moyen"

    print(f"\nCLUSTER {cluster_id} ({len(cluster_data)} clients)")
    print(f"   • Âge moyen : {age_mean:.1f} ans ({age_cat})")
    print(f"   • Revenu moyen : ${income_mean:.1f}k ({income_cat})")
    print(f"   • Score de dépense : {spending_mean:.1f} ({spending_cat})")
    print(f"   • Hommes : {male_percent:.1f}%")

    if income_mean < 40 and spending_mean < 40:
        profile = "Clients prudents"
        recommendations = ["Promotions", "Réductions", "Produits essentiels"]
    elif income_mean > 70 and spending_mean > 60:
        profile = "Clients premium"
        recommendations = ["VIP", "Luxe", "Expériences exclusives"]
    elif income_mean > 70 and spending_mean < 40:
        profile = "Fort potentiel"
        recommendations = ["Offres personnalisées", "Upselling"]
    elif age_mean < 35 and spending_mean > 60:
        profile = "Jeunes dépensiers"
        recommendations = ["Digital marketing", "Influence"]
    else:
        profile = "Clients standards"
        recommendations = ["Fidélité", "Cross-selling"]

    print(f"   → Profil : {profile}")
    print(f"   → Actions : {', '.join(recommendations)}")


# ============================================================
# 3. STANDARDISATION UNIQUE (clé de la cohérence)
# ============================================================
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_original[features])

labels = df['Cluster'].values
# ============================================================
# 4. VALIDATION AVEC SILHOUETTE
# ============================================================
print("\n" + "="*60)
print("2. VALIDATION DU CLUSTERING")
print("="*60)

sil_score = silhouette_score(X_scaled, labels)
print(f"\nSilhouette Score pour K=4 : {sil_score:.3f}")

# Interprétation du score
if sil_score > 0.7:
    print("Très bien séparé")
elif sil_score > 0.5:
    print("Correct / bien séparé")
elif sil_score > 0.25:
    print("Acceptable / chevauchement modéré")
else:
    print("Mauvaise affectation")


print("\nAnalyse détaillée par cluster :")
silhouette_vals = silhouette_samples(X_scaled, labels)

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for i in np.unique(labels):
    vals = silhouette_vals[labels == i]
    vals.sort()

    size = vals.shape[0]
    y_upper = y_lower + size

    color = plt.cm.tab10(float(i) / len(np.unique(labels)))
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper + 10

ax.axvline(x=sil_score, color="red", linestyle="--")
ax.set_title("Silhouette Plot")
ax.set_xlabel("Coefficient")
ax.set_yticks([])
plt.tight_layout()
plt.show()


# ============================================================
# 5. COMPARAISON DES DIFFÉRENTS K
# ============================================================
print("\n" + "="*60)
print("3. ANALYSE COMPARATIVE DU NOMBRE DE CLUSTERS")
print("="*60)

k_values = [2, 3, 4, 5, 6]
inertia_values = []
silhouette_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(X_scaled)

    inertia_values.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels_k)
    silhouette_values.append(sil)


# Graphiques
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_values, inertia_values, 'bo-', linewidth=2)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertie")

axes[1].plot(k_values, silhouette_values, 'ro-', linewidth=2)
axes[1].set_title("Silhouette par K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.show()


print("\nComparaison :")
print("-"*40)
for i, k in enumerate(k_values):
    print(f"K = {k} : Inertie = {inertia_values[i]:.2f}, Silhouette = {silhouette_values[i]:.3f}")

print("\nTâche terminée avec succès !")
