#!/bin/bash

echo "------------------------------------------------"
echo "Menjalankan Pipeline Pelatihan (MKN-2 VLM)"
echo "------------------------------------------------"

# 1. Validasi & Pembersihan Gambar (Optional tapi disarankan)
echo "[1/3] Validasi integritas gambar..."
python src/preprocess.py

# 2. Pembagian Data (Train, Val, Test) -> KRUSIAL!
echo "[2/3] Membagi dataset (80% Train, 10% Val, 10% Test)..."
python src/data_splitter.py

# 3. Eksekusi Training Qwen2.5-VL
echo "[3/3] Memulai proses Fine-Tuning (QLoRA)..."
python -m src.training.train

echo "------------------------------------------------"
echo "Training Selesai! Adapter disimpan di models/final_adapter"
echo "------------------------------------------------"