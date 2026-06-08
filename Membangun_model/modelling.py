import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_baseline():
    # Memuat data hasil preprocessing (fitur dan target secara terpisah)
    X_train = pd.read_csv('preprocessing/namadataset_preprocessing/train_preprocessed.csv')
    y_train = pd.read_csv('preprocessing/namadataset_preprocessing/y_train.csv').squeeze('columns')
    X_test = pd.read_csv('preprocessing/namadataset_preprocessing/test_preprocessed.csv')
    y_test = pd.read_csv('preprocessing/namadataset_preprocessing/y_test.csv').squeeze('columns')
    
    # Setup MLflow Autolog lokal
    mlflow.set_experiment("Diabetes_Baseline_Autolog")
    mlflow.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        print(f"Baseline Autolog Accuracy: {accuracy_score(y_test, preds):.4f}")

if __name__ == '__main__':
    train_baseline()
