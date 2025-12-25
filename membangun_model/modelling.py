import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import dagshub

def main():
    # 1. Init DagsHub
    # Pastikan repo_owner dan repo_name sesuai dengan DagsHub Anda
    dagshub.init(repo_owner='psribuezraa', repo_name='Telco-Customer-Churn', mlflow=True)
    mlflow.set_experiment("Telco-Churn-Autolog-Final") 

    # 2. Load Data
    # Menggunakan try-except agar path aman baik di Laptop maupun di GitHub Actions
    print("Memuat data...")
    try:
        # Path untuk GitHub Actions
        df = pd.read_csv("membangun_model/dataset_preprocessing/train.csv")
    except FileNotFoundError:
        # Path untuk Run Lokal (Laptop)
        df = pd.read_csv("dataset_preprocessing/train.csv")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Aktifkan Autolog (INI YANG DIMINTA REVIEWER)
    # log_models=True artinya model akan otomatis disimpan tanpa perlu kode tambahan
    print("Mengaktifkan Autolog...")
    mlflow.sklearn.autolog(log_models=True)

    # 4. Training
    with mlflow.start_run():
        print("Sedang training model...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Kita TIDAK PERLU lagi melakukan log_metric atau save model manual.
        # Autolog akan menangkap akurasi, precision, recall, dan file model secara otomatis.
        
        print("Training Selesai! Model dan metrik telah direkam oleh Autolog.")

if __name__ == "__main__":
    main()