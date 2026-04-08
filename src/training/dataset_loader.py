import torch
from torch.utils.data import Dataset
from PIL import Image

class RutilahuDataset(Dataset):
    def __init__(self, json_data, processor):
        self.json_data = json_data
        self.processor = processor

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data[idx]
        # Mengambil gambar tampak luar (ext) dan dalam (int) 
        images = [Image.open(p).convert("RGB") for p in item["images"]]
        
        # Format pesan sesuai standar 
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Foto 1 (tampak luar):"},
                    {"type": "image"},
                    {"type": "text", "text": "Foto 2 (tampak dalam):"},
                    {"type": "image"},
                    {"type": "text", "text": "Analisis material & kondisi sesuai aturan agregasi MKN-2."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["target_json"]}]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        
        return {k: v.squeeze(0) for k, v in inputs.items()}