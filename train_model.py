#Script pour entraîner les modèles de machine learning.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    # Chargement des données
    data = load_data('processed_car_insurance.csv')

    # Séparation des données
    X_train, X_test, y_train, y_test = split_data(data, 'target_column_name')

    # Entraînement du modèle
    model = train_model(X_train, y_train)

    # Évaluation du modèle
    predictions = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')

    # Sauvegarde du modèle
    save_model(model, 'final_model.joblib')
