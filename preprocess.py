import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

label_encoder = LabelEncoder() 

def load_data(filepath):
    df = pd.read_csv(filepath)
    # print(df.info())
    # print(df.head())
    return df

def examine_data(df):
    print("Taille du jeu de données:", df.shape)
    print("Types de données:\n", df.dtypes)
    print("Valeurs manquantes:\n", df.isna().sum())

    # Visualiser la distribution des données
    df.hist(bins=30, figsize=(20, 15))
    plt.show()
    
def label_encode_columns(df, cols):
    df_copy = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_copy[col] = label_encoder.fit_transform(df_copy[col])
    return df_copy

def preprocess_data(df):
    # Séparer les variables en numériques et catégorielles
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.pop()
  
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Pipeline pour les variables numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputation par la médiane
        ('scaler', StandardScaler())  # Normalisation des données
    ])

    # Transformer function for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('label_encoder', FunctionTransformer(lambda df: label_encode_columns(df, categorical_cols), validate=False)),  # Encodage Label
        ('scaler', StandardScaler())  # Normalisation des données
    ])

    # Préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Préparation des données
    df_processed = pd.DataFrame(preprocessor.fit_transform(df))
    df_processed.columns = numerical_cols+categorical_cols
    df_outcome = pd.DataFrame(df['outcome'])

    df_processed = pd.concat([df_processed, df_outcome], axis=1)
    print(df_processed)



    # Convert the result back to a DataFrame
    return df_processed

def threshold_cleaner(data,col, threshold):
     for i in range(data[col].shape[0]):
        if data[col][i] > threshold :
            data.loc[i, col] = None
            print(data[col][i])

def save_processed_data(df, filename):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Chemin vers le fichier de données
    data_path = 'car_insurance.csv'
    processed_data_path = 'processed_car_insurance.csv'

    # Charger et examiner les données
    data = load_data(data_path)
    data = data.drop('id',axis=1)

    threshold_cleaner(data,'children',1)
    threshold_cleaner(data,'speeding_violations',30)

    # Préparer les données
    processed_data = preprocess_data(data)


    # Enregistrer les données transformées
    save_processed_data(processed_data, processed_data_path)
