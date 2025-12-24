import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import dagshub
import joblib  # <--- Kita pakai ini untuk simpan manual

def main():
    dagshub.init(repo_owner='psribuezraa', repo_name='Telco-Customer-Churn', mlflow=True)
    mlflow.set_experiment("Telco-Churn-Fix-Final-2") 

    print("1. Memuat data...")
    try:
        df = pd.read_csv("membangun_model/dataset_preprocessing/train.csv")
    except FileNotFoundError:
        df = pd.read_csv("dataset_preprocessing/train.csv")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Mulai Training...")
    # Kita matikan autolog sebentar biar tidak pusing, kita log manual saja
    # mlflow.autolog() 

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Log Metrik
        y_pred = clf.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        
        # --- CARA PAKSA (PLAN B) ---
        print("3. Menyimpan file model.pkl di laptop...")
        joblib.dump(clf, "model.pkl") # Simpan di folder lokal dulu
        
        print("4. Mengupload model.pkl ke DagsHub...")
        mlflow.log_artifact("model.pkl") # Upload file fisik tersebut
        
        print("SELESAI! File 'model.pkl' berhasil dikirim.")

if __name__ == "__main__":
    main()