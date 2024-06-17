import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def train_best_model(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier()
    }

    params = {
        'LogisticRegression': {'model__C': [0.1, 1, 10, 100]},
        'RandomForest': {'model__n_estimators': [10, 50, 100]},
        'SVC': {'model__C': [0.1, 1, 10], 'model__gamma': [0.001, 0.01, 0.1]},
        'KNN': {'model__n_neighbors': [3, 5, 7]}
    }

    best_models = {}
    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        grid_search = GridSearchCV(pipe, params[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best Parameters for {name}: {grid_search.best_params_}")

    return best_models

def save_model(model, filename):
    joblib.dump(model, filename)

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Chargement des données
    data_path = 'processed_car_insurance.csv'
    data = pd.read_csv(data_path)


    # Séparation des données
    target_column_index = data.shape[1]-1 # Remplacez par l'index réel de la colonne cible
    X_train, X_test, y_train, y_test = split_data(data, target_column_index)

    # Entraînement des modèles et sélection du meilleur modèle
    best_models = train_best_model(X_train, y_train)

    for name, model in best_models.items():
        print(f"\nÉvaluation du modèle: {name}")
        predictions = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
        print("Classification Report:\n", classification_report(y_test, predictions))
        # Sauvegarde du modèle
        save_model(model, f'final_model_{name}.joblib')
