import pandas as pd
import numpy as np
import os

# pip install pandas numpy scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- KONFIGURASI PATH ---
# Menggunakan path relatif agar aman saat dijalankan di GitHub Actions nanti
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, './') # Simpan di folder preprocessing

def load_data(path):
    """Fungsi untuk memuat data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {path}")
    
    df = pd.read_csv(path)
    print(f"[INFO] Dataset dimuat: {df.shape}")
    return df

def clean_data(df):
    """Fungsi untuk membersihkan data mentah."""
    df = df.copy()
    
    # 1. Perbaikan tipe data TotalCharges (String -> Float)
    # 'coerce' akan mengubah string kosong menjadi NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 2. Isi missing values (akibat spasi kosong tadi) dengan 0
    df['TotalCharges'].fillna(0, inplace=True)
    
    # 3. Hapus customerID (tidak berguna untuk model)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # 4. Encoding Target (Yes/No -> 1/0)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    print("[INFO] Data Cleaning selesai.")
    return df

def get_preprocessor(X):
    """Membuat Pipeline Preprocessing."""
    # Identifikasi kolom otomatis
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"[INFO] Fitur Numerik: {len(numerical_cols)}")
    print(f"[INFO] Fitur Kategorikal: {len(categorical_cols)}")

    # Pipeline Numerik
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Pipeline Kategorikal
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Gabungkan
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], verbose_feature_names_out=False) # Agar nama kolom hasil tetap bersih
    
    return preprocessor

def main():
    # 1. Load Data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. Clean Data
    df_clean = clean_data(df)

    # 3. Split Data
    # PENTING: Split dilakukan SEBELUM preprocessing agar tidak ada data leakage
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Preprocessing
    preprocessor = get_preprocessor(X_train)

    # FIT hanya pada TRAIN, Transform pada keduanya
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Kembalikan ke format DataFrame agar mudah dibaca manusia/disimpan
    feature_names = preprocessor.get_feature_names_out()
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # Gabungkan kembali dengan label/target
    train_final = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test_final = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    # 5. Simpan Hasil
    train_final.to_csv(os.path.join(OUTPUT_PATH, 'train.csv'), index=False)
    test_final.to_csv(os.path.join(OUTPUT_PATH, 'test.csv'), index=False)

    print(f"[SUCCESS] Preprocessing selesai!")
    print(f"Data Train tersimpan: {train_final.shape}")
    print(f"Data Test tersimpan:  {test_final.shape}")

if __name__ == "__main__":
    main()