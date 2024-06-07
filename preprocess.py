import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Effectue le prétraitement des données, y compris le traitement des valeurs manquantes, l'encodage, et la normalisation."""
    # Traitement des valeurs manquantes
    for col in df.columns:
        if df[col].isna().sum() > len(df) * 0.33:
            df.drop(col, axis=1, inplace=True)  # Suppression de la colonne si trop de NA
        elif df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)  # Remplacement par la médiane pour les numériques
        else:
            most_frequent = df[col].mode()[0]
            df[col].fillna(most_frequent, inplace=True)  # Remplacement par le mode pour les catégorielles

    # Gestion des valeurs aberrantes
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        q_low = df[col].quantile(0.01)
        q_hi  = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q_service, upper=q_hi)

    # Encodage des variables catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].unique()) == 2:
            df[col] = LabelEncoder().fit_transform(df[col])  # Encodage pour les booléens
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)  # Encodage One-Hot

    # Normalisation des variables numériques
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def analyze_correlations(df):
    """Affiche la matrice de corrélations et retourne les variables significatives."""
    correlation_matrix = df.corr()
    print("Matrice de corrélations:\n", correlation_matrix)

    # Affichage des corrélations importantes (exemple avec seuil arbitraire > 0.5 en valeur absolue)
    significant_features = correlation_matrix.columns[correlation_matrix.abs().max() > 0.5].tolist()
    return significant_features

def visualize_correlations(df, variables):
    """Visualise les relations entre les variables sélectionnées."""
    scatter_matrix(df[variables], alpha=0.2, figsize=(10, 10), diagonal='kde')
    plt.show()

def save_processed_data(df, filename):
    """Sauvegarde les données traitées dans un fichier CSV."""
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    data = load_data('car_insurance.csv')
    processed_data = preprocess_data(data)
    save_processed_data(processed_data, 'processed_car_insurance.csv')

    # Analyse des corrélations
    features_to_visualize = analyze_correlations(processed_data)
    visualize_correlations(processed_data, features_to_visualize)
