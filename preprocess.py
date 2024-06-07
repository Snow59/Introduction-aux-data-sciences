import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(df.info())
    print(df.head())
    return df

def examine_data(df):
    print("Taille du jeu de données:", df.shape)
    print("Types de données:\n", df.dtypes)
    print("Valeurs manquantes:\n", df.isna().sum())

    # Visualiser la distribution des données
    df.hist(bins=30, figsize=(20, 15))
    plt.show()

def preprocess_data(df):
    # Séparer les variables en numériques et catégorielles
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Pipeline pour les variables numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputation par la médiane
        ('scaler', StandardScaler())  # Normalisation des données
    ])

    # Pipeline pour les variables catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation par le mode
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encodage One-Hot
    ])

    # Préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Préparation des données
    df_processed = preprocessor.fit_transform(df)
    return pd.DataFrame(df_processed)

def save_processed_data(df, filename):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Chemin vers le fichier de données
    data_path = 'car_insurance.csv'
    processed_data_path = 'processed_car_insurance.csv'

    # Charger et examiner les données
    data = load_data(data_path)
    examine_data(data)

    # Préparer les données
    processed_data = preprocess_data(data)

    # Enregistrer les données transformées
    save_processed_data(processed_data, processed_data_path)
