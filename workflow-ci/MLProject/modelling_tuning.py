import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub  # <--- Library Wajib untuk Advanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONFIGURASI ---
TRAIN_PATH = 'dataset_preprocessing/train.csv'
TEST_PATH = 'dataset_preprocessing/test.csv'

# --- LOAD DATA ---
def load_data(path):
    df = pd.read_csv(path)
    # Asumsi kolom terakhir adalah Target 'Churn'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def main():
    # 1. Setup DagsHub Connection (Advanced)
    print("[INFO] Menghubungkan ke DagsHub...")
    
    # Ini akan otomatis setup MLflow Tracking URI ke DagsHub
    dagshub.init(repo_owner="psribuezraa", repo_name="Telco-Customer-Churn", mlflow=True)
    
    mlflow.set_experiment("Telco-Churn-Tuning-Online") # Nama eksperimen di DagsHub

    # 2. Load Data
    print("[INFO] Memuat data...")
    try:
        X_train, y_train = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 3. Mulai MLflow Run
    with mlflow.start_run():
        print("[INFO] Memulai Hyperparameter Tuning...")

        # --- A. DEFINISI GRID PARAMETER ---
        # Sederhanakan sedikit agar running tidak terlalu lama saat testing koneksi
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2]
        }

        # --- B. GRID SEARCH ---
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Ambil hasil terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Parameter Terbaik: {best_params}")

        # --- C. MANUAL LOGGING (Syarat Skilled/Advanced) ---
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        mlflow.log_param("model_type", "RandomForest_GridSearch_DagsHub")

        # --- D. EVALUASI ---
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Akurasi: {acc:.4f}")

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- E. ARTIFACTS (Confusion Matrix) ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plot_filename = "confusion_matrix_dagshub.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename) # Upload gambar ke DagsHub

        # Log Model
        mlflow.sklearn.log_model(best_model, "model")

        if os.path.exists(plot_filename):
            os.remove(plot_filename)

        print(f"[SUCCESS] Selesai! Cek hasil di: https://dagshub.com/{psribuezraa}/{Telco-Customer-Churn}.mlflow")

if __name__ == "__main__":
    main()