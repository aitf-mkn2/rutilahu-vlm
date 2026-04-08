import json
import os
import torch
from transformers import AutoProcessor, Trainer, TrainingArguments
from .model_config import setup_model_and_peft
from .dataset_loader import RutilahuDataset
# Mengimpor fungsi splitter baru
from src.data_splitter import get_three_split_data 

def main():
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    DATA_PATH = "data/dataset.json"

    # 1. Inisialisasi Processor & Model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = setup_model_and_peft(MODEL_ID)

    # 2. Load Dataset dengan 3-Way Split
    # Mengambil jatah 80% untuk belajar dan 10% untuk simulasi (validation)
    train_data = get_three_split_data(DATA_PATH, mode='train')
    val_data = get_three_split_data(DATA_PATH, mode='val')

    train_dataset = RutilahuDataset(train_data, processor)
    val_dataset = RutilahuDataset(val_data, processor)

    # 3. Training Arguments (Ditambah pengaturan evaluasi)
    training_args = TrainingArguments(
        output_dir="./output/checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        # Pengaturan agar model melakukan evaluasi setiap akhir epoch
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True, # Mengambil versi terbaik setelah latihan
        report_to="tensorboard"
    )

    # 4. Eksekusi Trainer dengan Validation Set
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # Penambahan data Try Out
    )

    print(f"Memulai Training dengan {len(train_data)} data...")
    trainer.train()
    
    # Simpan adapter akhir
    model.save_pretrained("./models/final_adapter")
    print("Training Selesai! Model terbaik disimpan di models/final_adapter")

if __name__ == "__main__":
    main()