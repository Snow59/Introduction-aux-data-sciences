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
    precision = precision_score(y_test, predictions, average='binary')  # adapt for multi-class if needed
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    report = classification_report(y_test, predictions)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", report)

    # Afficher les prédictions pour chaque échantillon (optionnel)
    for true, pred in zip(y_test, predictions):
        print(f"True class: {true}, Predicted class: {pred}")

def perform_cross_validation(model, X, y):
    cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cross_validator, scoring='accuracy')
    print("Cross-Validation Scores:", acc_scores)
    print("Mean Accuracy:", acc_scores.mean())
    print("Standard Deviation:", acc_scores.std())

def analyze_correlations(df, target_column):
    """Analyse les corrélations entre les variables et affiche les résultats."""
    corr_matrix = df.corr()
    print("\nCorrélations des variables :\n", corr_matrix)
    
    # Afficher les corrélations d'une variable particulière avec les autres
    target_corr = corr_matrix[target_column].sort_values(ascending=False)
    print(f"\nCorrélations avec la variable cible {target_column} :\n", target_corr)
    
    # Heatmap des corrélations
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Heatmap des Corrélations')
    plt.show()
    
    # Scatter matrix
    scatter_matrix(df, figsize=(15, 15), diagonal='kde')
    plt.show()

if __name__ == "__main__":
    test_data_path = 'processed_car_insurance.csv'
    model_path = 'final_model.joblib'
    
    # Chargement des données de test
    test_data = load_data(test_data_path)
    X_test = test_data.drop('target_column_name', axis=1)
    y_test = test_data['target_column_name']
    
    # Chargement du modèle
    model = load_model(model_path)
    
    # Évaluation du modèle sur les données de test
    evaluate_model(model, X_test, y_test)
    
    # Validation croisée sur l'ensemble des données
    perform_cross_validation(model, X_test, y_test)
    
    # Analyse des corrélations
    analyze_correlations(test_data, 'target_column_name')
