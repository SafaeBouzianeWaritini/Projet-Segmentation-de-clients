"""
Author: Imane Chaouqi
Role: Customer Segmentation - Clustering (KMeans)
Project: Customer Segmentation for Marketing Strategy
Goal: Segment customers to help marketing teams target different categories
"""

#  IMPORT DES LIBRAIRIES

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. CHARGEMENT DES DONNÉES
df = pd.read_csv("Mall_Customers_Cleaned.csv")

print("Aperçu des données :")
print(df.head())

# 2. SÉLECTION DES VARIABLES POUR LE CLUSTERING
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

print("\nVariables utilisées pour le clustering :")
print(X.describe())

# 3. MÉTHODE DU COUDE (ELBOW METHOD)
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Graphe Elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertia (Erreur intra-cluster)")
plt.title("Elbow Method - Choix du nombre optimal de clusters")
plt.grid(True)
plt.show()  

# 4. ENTRAÎNEMENT FINAL AVEC K = 4
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 5. VISUALISATION DES CLUSTERS
plt.figure(figsize=(8, 6))
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Segmentation des clients (K = 4)")
plt.colorbar(label="Cluster")
plt.show()   

# 6. ANALYSE DES CLUSTERS
cluster_analysis = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

print("\nAnalyse moyenne par cluster :")
print(cluster_analysis)

# 7. SAUVEGARDE
df.to_csv("Mall_Customers_With_Clusters.csv", index=False)
print("\nSuccès : fichier 'Mall_Customers_With_Clusters.csv' créé.")
