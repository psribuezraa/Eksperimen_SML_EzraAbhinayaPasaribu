import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
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
    print("[INFO] Menghubungkan ke DagsHub...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    
    # Set nama eksperimen berbeda agar mudah dibedakan di UI
    mlflow.set_experiment("Telco-Churn-Single-Model")

    print("Memuat data...")
    try:
        X_train, y_train = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal memuat data: {e}")
        return

    with mlflow.start_run():
        print("[INFO] Memulai Training Model (Single Run)...")

        # 1. Definisi Parameter (Bisa diubah manual di sini)
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }

        # 2. Latih Model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 3. Log Parameter
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest_Single")

        # 4. Evaluasi
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Akurasi: {acc:.4f}")

        # 5. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # 6. Confusion Matrix & Artifacts
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plot_filename = "confusion_matrix.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        
        # 7. Simpan Model
        mlflow.sklearn.log_model(model, "model")

        # Bersihkan file lokal
        if os.path.exists(plot_filename):
            os.remove(plot_filename)

        print(f"[SUCCESS] Selesai! Cek DagsHub: https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

if __name__ == "__main__":
    main()