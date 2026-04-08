import json
import random

def get_three_split_data(file_path, mode='train'):
    with open(file_path, 'r') as f:
        full_data = json.load(f)
    
    random.seed(42) 
    random.shuffle(full_data)
    
    # Menghitung titik potong (80% - 10% - 10%)
    n = len(full_data)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    if mode == 'train':
        return full_data[:train_end]     # 80% untuk latihan
    elif mode == 'val':
        return full_data[train_end:val_end] # 10% untuk Try Out (saat latihan)
    else:
        return full_data[val_end:]       # 10% sisanya untuk Ujian Akhir