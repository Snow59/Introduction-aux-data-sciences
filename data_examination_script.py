
import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV
df = pd.read_csv('car_insurance.csv')

# Affichage des premières lignes du DataFrame
print(df.head())

# Affichage des informations du DataFrame
print(df.info())

# Affichage des statistiques descriptives des données
print(df.describe())

# Comptage des valeurs manquantes
print(df.isna().sum())

# Histogrammes des variables numériques
df.hist(figsize=(12, 10))
plt.show()

# Boxplots pour détecter les valeurs aberrantes
df.boxplot(figsize=(12, 10))
plt.show()
