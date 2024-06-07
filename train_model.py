import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(filepath):
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(filepath)

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'apprentissage et de test."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, y_train):
    """Entraîne un modèle de régression logistique sur les données d'apprentissage."""
    model = LogisticRegression(max_iter=1000)  # Augmentation du nombre d'itérations si nécessaire
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test et affiche les résultats."""
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    # Chargement des données
    data_path = 'path/to/your/processed_data.csv'
    data = load_data(data_path)

    # Séparation des données
    X_train, X_test, y_train, y_test = split_data(data, 'target_column_name')  # Remplacez 'target_column_name'

    # Entraînement du modèle de régression logistique
    logistic_model = train_logistic_regression(X_train, y_train)

    # Évaluation du modèle
    evaluate_model(logistic_model, X_test, y_test)
