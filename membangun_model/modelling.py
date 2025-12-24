import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
TRAIN_PATH = 'dataset_preprocessing/train.csv' # Pastikan path ini sesuai
REPO_OWNER = "psribuezraa"
REPO_NAME = "Telco-Customer-Churn"

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'], errors='ignore') # Asumsi kolom target bernama 'Churn' atau index terakhir
    # Jika dataset Anda targetnya di kolom terakhir dan belum ada nama header:
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]
    
    # Sesuaikan dengan dataset Anda, pastikan X dan y terpisah benar
    y = df.iloc[:, -1] 
    X = df.iloc[:, :-1]
    return X, y

def main():
    # 1. Init Dagshub
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Telco-Churn-Autolog")

    # 2. Load Data
    print("Memuat data...")
    X, y = load_data(TRAIN_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. AKTIFKAN AUTOLOG (Ini yang diminta Reviewer)
    # log_models=True agar model.pkl dan estimator.html tersimpan otomatis
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run():
        print("Melatih model dengan Autolog...")
        
        # Training biasa (tanpa log manual)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi (Autolog akan otomatis mencatat akurasi dll)
        print("Model training selesai.")

if __name__ == "__main__":
    main()