import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def load_data(filepath):
    return pd.read_csv(filepath, header=None)  # Charger sans en-têtes

def check_and_discretize_target(y):
    # Vérifier le type de valeurs dans la colonne cible
    if y.dtype.kind in 'fc':  # f: float, c: complex
        print("Les valeurs de la colonne cible sont continues. Conversion en classes discrètes...")
        # Discrétiser les valeurs continues en classes discrètes en gérant les duplications
        y = pd.qcut(y.rank(method='first'), q=2, labels=[0, 1], duplicates='drop')
    return y

def split_data(df, target_column_index, test_size=0.2, random_state=42):
    X = df.drop(target_column_index, axis=1)
    y = df[target_column_index]
    y = check_and_discretize_target(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, y_train):
    # Paramètres pour la recherche de grille
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [200, 300, 500]
    }

    # Initialiser le modèle de régression logistique
    log_reg = LogisticRegression()

    # Initialiser GridSearchCV avec validation croisée
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Entraîner le modèle avec la recherche de grille
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")

    # Retourner le meilleur modèle trouvé
    return grid_search.best_estimator_

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    # Chargement des données
    data_path = 'processed_car_insurance.csv'
    data = load_data(data_path)

    # Séparation des données
    target_column_index = 2  
    X_train, X_test, y_train, y_test = split_data(data, target_column_index)

    # Entraînement du modèle de régression logistique
    logistic_model = train_logistic_regression(X_train, y_train)

    # Évaluation du modèle
    predictions = logistic_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    # Sauvegarde du modèle
    save_model(logistic_model, 'final_model.joblib')
