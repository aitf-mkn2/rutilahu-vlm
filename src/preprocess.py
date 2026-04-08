import json
import os
from PIL import Image

def clean_dataset(input_file, image_folder):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    valid_data = []
    print(f"Memvalidasi {len(data)} entri data...")

    for entry in data:
        # Pastikan ada 2 path gambar
        if len(entry["images"]) == 2:
            all_exist = True
            for img_path in entry["images"]:
                if not os.path.exists(img_path):
                    all_exist = False
                    break
            
            if all_exist:
                valid_data.append(entry)

    # Simpan data yang sudah valid
    with open("data/cleaned_dataset.json", 'w') as f:
        json.dump(valid_data, f, indent=4)
    
    print(f"Selesai. Data valid: {len(valid_data)}")

if __name__ == "__main__":
    clean_dataset("data/dataset.json", "data/images/")