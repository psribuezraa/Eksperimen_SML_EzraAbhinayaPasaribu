import time
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Summary, Counter, Gauge, Histogram
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

# --- 1. DEFINISI METRICS (Agar Mencapai Target Advanced) ---

# A. Metrics Sistem
REQUEST_COUNT = Counter('app_request_count', 'Total Request HTTP yang masuk')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Waktu yang dibutuhkan untuk memproses request')

# B. Metrics Model/Bisnis
PREDICTION_COUNT = Counter('model_prediction_count', 'Total Prediksi yang dilakukan')
CHURN_PREDICTED = Counter('model_churn_predicted_count', 'Total prediksi Churn=Yes')
NO_CHURN_PREDICTED = Counter('model_no_churn_predicted_count', 'Total prediksi Churn=No')

# C. Metrics Distribusi Data (Untuk deteksi Data Drift)
INPUT_TENURE = Histogram('data_input_tenure', 'Distribusi Tenure user')
INPUT_CHARGES = Histogram('data_input_monthly_charges', 'Distribusi Monthly Charges')
PROBABILITY_CONFIDENCE = Gauge('model_confidence_score', 'Probability confidence dari prediksi terakhir')

# --- 2. LOAD MODEL DUMMY (Simulasi) ---
# Kita gunakan model dummy agar script ini bisa jalan mandiri tanpa perlu load pickle yang rumit
print("Melatih model dummy untuk serving...")
from sklearn.datasets import make_classification
X_dummy, y_dummy = make_classification(n_samples=100, n_features=19, random_state=42)
model = RandomForestClassifier()
model.fit(X_dummy, y_dummy)
print("Model siap serving!")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # Ambil data JSON dari request user
        data = request.json
        # Asumsi data masuk berupa list values: {'features': [0, 1, 2...]}
        features = np.array(data['features']).reshape(1, -1)
        
        # Lakukan Prediksi
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][np.argmax(model.predict_proba(features))]
        
        # --- UPDATE METRICS ---
        PREDICTION_COUNT.inc()
        PROBABILITY_CONFIDENCE.set(proba) # Gunakan Gauge untuk nilai yang naik turun
        
        # Simpan metrik input data (Contoh: kolom ke-5 tenure, ke-18 charges)
        # Kita ambil sample saja dari input features
        try:
            # Asumsi dummy indeks
            val_tenure = features[0][4] 
            val_charges = features[0][-1] 
            INPUT_TENURE.observe(val_tenure)
            INPUT_CHARGES.observe(val_charges)
        except:
            pass # Abaikan jika indeks out of bound

        if prediction == 1:
            CHURN_PREDICTED.inc()
        else:
            NO_CHURN_PREDICTED.inc()
            
        # Hitung latency
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(proba),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    # Jalankan server metrics Prometheus di port 8000
    start_http_server(8000)
    print("Prometheus Metrics berjalan di port 8000")
    
    # Jalankan server API Flask di port 5000
    app.run(host='0.0.0.0', port=5000)

# Pastikan Counter didefinisikan di atas
churn_count = Counter('model_churn_predicted_count_total', '...')
no_churn_count = Counter('model_no_churn_predicted_count_total', '...') # <--- INI WAJIB ADA

# Di dalam fungsi predict:
prediction = model.predict(data)
if prediction[0] == 1:
    churn_count.inc()
else:
    no_churn_count.inc() # <--- INI PENTING AGAR "NO DATA" HILANG