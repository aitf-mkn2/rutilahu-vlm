from datasets import load_dataset
from PIL import Image
import os


class VLMdataset:
    def __init__(self, dataset_name=None, data_path=None, split="train", base_path=""):
        """
        Load dataset untuk VLM.

        - dataset_name: nama dataset di HuggingFace
        - data_path: path file lokal (JSONL)
        - split: bagian data (train / validation / test)

        Jika dataset_name ada → load dari HuggingFace
        Jika tidak → load dari lokal
        """
         
        if dataset_name:
            self.dataset = load_dataset(dataset_name, split=split)
        else:
            self.dataset = load_dataset("json", data_files=data_path, split=split)

        self.base_path = base_path

    def __len__(self):
        return len(self.dataset)
    
    def _resolve_image_path(self, img_path):
        filename = os.path.basename(img_path)

        # mapping folder berdasarkan nama file
        if "multi" in filename:
            folder = "multi_images"
        elif "ext" in filename:
            folder = "single_images_exterior"
        elif "int" in filename:
            folder = "single_images_interior"
        else:
            raise ValueError(f"Tidak bisa menentukan folder untuk: {filename}")

        if self.base_path:
            return os.path.join(self.base_path, folder, filename)
        else:
            return os.path.join(folder, filename)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Ambil messages
        messages = sample["messages"]

        #  Parse USER 
        user_msg = next(m for m in messages if m["role"] == "user")
        contents = user_msg["content"]

        images = []
        instruction_parts = []

        for item in contents:
            if item["type"] == "image":
                img = item["image"]

                # handle path & HF image
                if isinstance(img, str):
                    img_path = self._resolve_image_path(img)
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        raise ValueError(f"Gagal load image: {img_path} | {e}")

                elif isinstance(img, Image.Image):
                    pass  

                else:
                    raise ValueError(f"Format image tidak dikenali: {type(img)}")

                images.append(img)

            elif item["type"] == "text":
                instruction_parts.append(item["text"])

        instruction = "\n".join(instruction_parts)

        #  Parse ASSISTANT (target output) 
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        output_text = ""

        for item in assistant_msg["content"]:
            if item["type"] == "text":
                output_text += item["text"]

        #  Validasi ringan 
        if len(images) == 0:
            raise ValueError(f"Sample {idx} tidak memiliki gambar")

        if output_text.strip() == "":
            raise ValueError(f"Sample {idx} tidak memiliki output")

        return {
            "images": images,             # list of PIL images
            "instruction": instruction,   # string
            "output": output_text.strip() # structured text
        }