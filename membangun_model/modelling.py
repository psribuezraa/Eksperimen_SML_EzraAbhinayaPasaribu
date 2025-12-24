import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

def main():
    # Setup DagsHub & MLflow
    dagshub.init(repo_owner='psribuezraa', repo_name='Telco-Customer-Churn', mlflow=True)
    mlflow.set_experiment("Telco-Churn-Final-Fix") # Ganti nama eksperimen biar bersih

    # Load Data (Pastikan path ini benar di laptop Anda)
    df = pd.read_csv("membangun_model/dataset_preprocessing/train.csv")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aktifkan Autolog
    mlflow.autolog()

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # Log Metrics Manual (Jaga-jaga)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        # --- INI YANG DIMINTA REVIEWER ---
        # Kita paksa simpan modelnya secara manual agar foldernya MUNCUL
        mlflow.sklearn.log_model(clf, "model_random_forest") 
        print("Model berhasil disimpan ke MLflow!")

if __name__ == "__main__":
    main()