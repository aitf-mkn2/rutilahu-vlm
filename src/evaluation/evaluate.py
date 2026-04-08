import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# Mengimpor fungsi splitter
from src.data_splitter import get_three_split_data 

def run_evaluation(image_ext, image_int, prompt):
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path = "./models/final_adapter"

    # 1. Load Model Dasar & Adapter
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    if os.path.exists(adapter_path):
        model.load_adapter(adapter_path)
        model.eval()
    else:
        print("Peringatan: Adapter tidak ditemukan, menggunakan model dasar!")
    
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Persiapan Pesan (Format Multi-Image sesuai Desain)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Foto 1 (tampak luar):"},
            {"type": "image", "image": image_ext},
            {"type": "text", "text": "Foto 2 (tampak dalam):"},
            {"type": "image", "image": image_int},
            {"type": "text", "text": prompt}
        ]
    }]

    # 3. Proses Input Visual
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    # 4. Inferensi (Tanpa Perhitungan Gradien)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        # Perbaikan: Memotong input agar hanya mengambil hasil jawaban AI
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return response[0]

if __name__ == "__main__":
    # Mengambil jatah data "Ujian Nasional" (10% Test)
    test_data = get_three_split_data("data/dataset.json", mode='test')
    
    if len(test_data) > 0:
        # Ambil satu contoh rumah dari data ujian
        example = test_data[0]
        # Sesuai urutan dataset: images[0] = luar, images[1] = dalam
        hasil = run_evaluation(
            example["images"][0], 
            example["images"][1], 
            "Analisis kondisi rumah ini dan berikan output dalam format JSON."
        )
        print("\n--- HASIL EVALUASI (DATA TEST) ---")
        print(hasil)
    else:
        print("Data test tidak tersedia.")