import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import dagshub

# 1. Inisialisasi DagsHub
dagshub.init(repo_owner="reginasitumorang826", repo_name="Membangun_model", mlflow=True)

# 2. Memuat Data Sesuai Folder Lokal Kriteria 2 (Wajib untuk Dicoding!)
train_df = pd.read_csv('namadataset_preprocessing/train_preprocessed.csv')
test_df = pd.read_csv('namadataset_preprocessing/test_preprocessed.csv')

# Memisahkan Fitur (X) and Target (y)
X_train = train_df.drop(columns=['Outcome'])
y_train = train_df['Outcome']
X_test = test_df.drop(columns=['Outcome'])
y_test = test_df['Outcome']

# 3. Menetapkan Nama Eksperimen MLflow
mlflow.set_experiment("Diabetes_Tuning_Model")

with mlflow.start_run():
    # 4. Hyperparameter Tuning dengan Grid Search
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 5. Menghitung Metrik Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 6. MANUAL LOGGING PARAMETER & METRIK
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # 7. LOGGING 2 ARTEFAK TAMBAHAN
    os.makedirs('plots', exist_ok=True)

    # Artefak 1: Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title('Confusion Matrix - Tuned Model')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    mlflow.log_artifact('plots/confusion_matrix.png', 'visualizations')

    # Artefak 2: Feature Importance
    plt.figure(figsize=(8,5))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    mlflow.log_artifact('plots/feature_importance.png', 'visualizations')

    # Log Model Akhir
    mlflow.sklearn.log_model(best_model, "tuned_rf_model")

    print(f"Tuning Model Berhasil Dilatih! Best Accuracy: {acc:.4f}")
