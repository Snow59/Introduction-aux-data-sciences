#Script pour évaluer les modèles sur les données de test et générer des métriques de performance.
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(filepath):
    return pd.read_csv(filepath)

def load_model(model_path):
    return jobl.load(model_path)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    # Chemin vers les données et le modèle
    test_data_path = 'processed_car_insurance_test.csv'
    model_path = 'final_model.joblib'
    
    # Chargement des données de test
    test_data = load_data(test_data_path)
    X_test = test_data.drop('target_column_name', axis=1)
    y_test = test_data['target_column_name']
    
    # Chargement du modèle
    model = load_model(model_path)
    
    # Évaluation du modèle
    evaluate_model(model, X_test, y_test)
