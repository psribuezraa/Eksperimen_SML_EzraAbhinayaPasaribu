# Gunakan Python versi ringan
FROM python:3.9-slim

# Set folder kerja dalam container
WORKDIR /app

# Copy daftar library dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode ke dalam container
COPY . .

# Buka port untuk API (5000) dan Metrics (8000)
EXPOSE 5000 8000

# Jalankan script
CMD ["python", "prometheus_exporter.py"]