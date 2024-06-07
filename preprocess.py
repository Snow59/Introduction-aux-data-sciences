import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """Importe les données depuis un fichier CSV."""
    df = pd.read_csv(filepath)
    print("Premières lignes des données importées :")
    print(df.head())
    print("\nInformations sur les données :")
    print(df.info())
    return df

def examine_data(df):
    """Examine les données pour en comprendre la structure."""
    print("\nTaille du jeu de données :", df.shape)
    print("\nTypes des données :")
    print(df.dtypes)
    print("\nValeurs manquantes par colonne :")
    print(df.isna().sum())
    print("\nDescription des données :")
    print(df.describe())
    df.hist(figsize=(20, 15))
    plt.show()

def preprocess_data(df):
    """Prépare les données pour les algorithmes de classification."""
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
    df_processed = pd.DataFrame(df_processed)  # Conversion en DataFrame
    return df_processed

def save_processed_data(df, filename):
    """Enregistre les données prétraitées dans un fichier CSV."""
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Chemin vers le fichier de données
    data_path = 'car_insurance.csv'
    processed_data_path = 'processed_car_insurance.csv'

    # Charger les données
    data = load_data(data_path)

    # Examiner les données
    examine_data(data)

    # Préparer les données
    processed_data = preprocess_data(data)

    # Enregistrer les données transformées
    save_processed_data(processed_data, processed_data_path)
