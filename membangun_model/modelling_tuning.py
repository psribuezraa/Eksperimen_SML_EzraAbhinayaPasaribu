import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONFIGURASI ---
TRAIN_PATH = 'dataset_preprocessing/train.csv'
TEST_PATH = 'dataset_preprocessing/test.csv'
REPO_OWNER = "psribuezraa"
REPO_NAME = "Telco-Customer-Churn"

def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def main():
    print("[INFO] Menghubungkan ke DagsHub untuk Tuning...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    
    # Nama eksperimen khusus untuk tuning
    mlflow.set_experiment("Telco-Churn-Hyperparameter-Tuning")

    print("Memuat data...")
    try:
        X_train, y_train = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    with mlflow.start_run():
        print("[INFO] Memulai Grid Search (Ini mungkin memakan waktu)...")

        # 1. Definisi Ruang Pencarian (Grid)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5]
        }

        # 2. Proses Grid Search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Ambil hasil terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Parameter Terbaik: {best_params}")

        # 3. Log Parameter Terbaik ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest_GridSearch")

        # 4. Evaluasi Model Terbaik
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Akurasi Terbaik: {acc:.4f}")

        # 5. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # 6. Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Tuned Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plot_filename = "confusion_matrix_tuned.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)

        # 7. Simpan Model
        mlflow.sklearn.log_model(best_model, "model_tuned")

        if os.path.exists(plot_filename):
            os.remove(plot_filename)

        print(f"[SUCCESS] Tuning Selesai! Cek DagsHub: https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

if __name__ == "__main__":
    main()