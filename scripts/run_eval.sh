#!/bin/bash

echo "------------------------------------------------"
echo "Menjalankan Pipeline Evaluasi (Data Test)"
echo "------------------------------------------------"

# 1. Cek keberadaan Adapter hasil latihan
if [ ! -d "./models/final_adapter" ]; then
    echo "ERROR: File './models/final_adapter' tidak ditemukan!"
    echo "Pesan: Kamu harus menyelesaikan 'run_train.sh' terlebih dahulu."
    exit 1
fi

# 2. Eksekusi Analisis pada 10% data "suci"
echo "Memproses inferensi menggunakan data ujian akhir..."
python -m src.evaluation.evaluate

echo "------------------------------------------------"
echo "Evaluasi Selesai. Periksa output JSON di terminal."
echo "------------------------------------------------"