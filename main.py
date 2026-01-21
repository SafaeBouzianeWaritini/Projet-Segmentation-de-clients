import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Chargement des données
df = pd.read_csv('Mall_Customers.csv')

# 2. Encodage de la variable 'Gender' (Transformation)
# On transforme 'Male'/'Female' en 1/0 pour que l'IA puisse lire la colonne
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# 3. Normalisation (Scaling)
# On sélectionne les colonnes numériques (on exclut l'ID du calcul mathématique)
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()

# On crée une copie pour ne pas écraser les données originales
df_final = df.copy()
df_final[features] = scaler.fit_transform(df[features])

# 4. Vérification finale
print("--- Aperçu des données prêtes pour l'étudiant 2 ---")
print(df_final.head())
print("\nStatistiques après normalisation (Moyenne proche de 0) :")
print(df_final[features].mean())

# 5. SAUVEGARDE
# On crée un nouveau fichier pour l'étape suivante
df_final.to_csv('Mall_Customers_Cleaned.csv', index=False)
print("\nSuccès : Le fichier 'Mall_Customers_Cleaned.csv' a été créé !")