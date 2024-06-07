import pandas as pd
from sklearn.model_selection import train_test_split
import joblib  # Si nécessaire pour sauvegarder les jeux

def load_processed_data(filepath):
    """Charge les données prétraitées depuis un fichier CSV."""
    return pd.read_csv(filepath)

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'apprentissage et de test."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test):
    """Optionnel: sauvegarde les jeux divisés en fichiers pour une utilisation ultérieure."""
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

if __name__ == "__main__":
    data = load_processed_data('path/to/your/processed_data.csv')
    X_train, X_test, y_train, y_hair = split_data(data, 'target_column_name')  # Remplace 'target_column_name' par le nom de ta colonne cible
    
    # Affiche des informations sur les jeux d'apprentissage et de test
    print(f"Total dataset size: {data.shape[0]} samples")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Proportion of training set: {100 * X_train.shape[0] / data.shape[0]:.2f}%")
    print(f"Proportion of test set: {100 * X_test.shape[0] / data.shape[0]:.2f}%")
    
    # Optionnel: Sauvegarde des jeux
    save_data(X_train, X_test, y_train, y_test)
