import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    """Charge les données à partir du fichier CSV spécifié."""
    return pd.read_csv(filepath)

def split_data(df, target_column):
    """Sépare les données en features et target, puis en ensembles d'entraînement et de test."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Entraîne un modèle RandomForest avec les données fournies."""
    model = RandomForestClassifier(n_estimators=100, random_type=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Sauvegarde le modèle entraîné dans un fichier joblib."""
    joblib.dump(model, filename)

def load_model(filename):
    """Charge un modèle depuis un fichier joblib."""
    return joblib.load(filename)

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

    # Chargement et vérification du modèle
    loaded_model = load_model('final_model.joblib')
    reloaded_predictions = loaded_model.predict(X_test)
    print(f'Reloaded Model Accuracy: {accuracy_profile(y_test, reloaded_predictions)}')
