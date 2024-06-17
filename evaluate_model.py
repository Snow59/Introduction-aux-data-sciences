import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def load_data(filepath):
    return pd.read_csv(filepath)

def load_model(model_path):
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    report = classification_report(y_test, predictions)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", report)

    for true, pred in zip(y_test, predictions):
        print(f"True class: {true}, Predicted class: {pred}")

def perform_cross_validation(model, X, y):
    cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cross_validator, scoring='accuracy')
    print("Cross-Validation Scores:", acc_scores)
    print("Mean Accuracy:", acc_scores.mean())
    print("Standard Deviation:", acc_scores.std())

def analyze_correlations(df, target_column_index):
    corr_matrix = df.corr()
    print("\nCorrélations des variables :\n", corr_matrix)

    target_corr = corr_matrix.iloc[:, target_column_index].sort_values(ascending=False)
    print(f"\nCorrélations avec la variable cible à l'index {target_column_index} :\n", target_corr)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Heatmap des Corrélations')
    plt.show()

    scatter_matrix(df, figsize=(15, 15), diagonal='kde')
    plt.show()

if __name__ == "__main__":
    test_data_path = 'processed_car_insurance.csv'
    model_path = 'final_model.joblib'

    # Chargement des données de test
    test_data =   pd.read_csv(test_data_path)

    # Afficher les colonnes disponibles pour vérifier le nom de la colonne cible
    print("Colonnes disponibles dans les données :")
    print(test_data.columns.tolist())

    # Index correct de la colonne cible
    target_column_index = test_data.shape[1]-1  # Remplacez par l'index réel de la colonne cible après vérification

    # Assurez-vous que l'index de la colonne cible est correct
    if target_column_index >= len(test_data.columns):
        raise IndexError(f"L'index de la colonne cible '{target_column_index}' n'existe pas dans les données.")

    target_column = test_data.columns[target_column_index]
    X_test = test_data.drop('outcome', axis=1)
    y_test = test_data['outcome']


    # Chargement du modèle
    model = load_model(model_path)

    # Évaluation du modèle sur les données de test
    print("Évaluation sur le jeu de test :")
    evaluate_model(model, X_test, y_test)

    # Validation croisée sur l'ensemble des données
    print("Validation croisée :")
    perform_cross_validation(model, X_test, y_test)

    # Analyse des corrélations
    print("Analyse des corrélations :")
    analyze_correlations(test_data, target_column_index)
