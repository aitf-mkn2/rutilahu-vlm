from datasets import load_dataset
from PIL import Image
import os
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download


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
            filename_map = {
                "train": "data/sft/train.jsonl",
                "validation": "data/sft/val.jsonl",
                "test": "data/sft/val.jsonl"   # nanti ganti pake test
            }

            if split not in filename_map:
                raise ValueError(f"Split tidak dikenali: {split}")

            file_path = hf_hub_download(
                repo_id=dataset_name,
                filename=filename_map[split],
                repo_type="dataset"
            )

            self.dataset = load_dataset(
                "json",
                data_files={"train": file_path},
                split="train"
            )
            
        elif data_path:
            self.dataset = load_dataset("json", data_files=data_path, split=split)
        else:
            raise ValueError("Harus isi dataset_name atau data_path")
        
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.image_cache = {}

    def __len__(self):
        return len(self.dataset)
    
    def _resolve_image_path(self, img_path):
        """
        Handle berbagai format path:
        - Windows path (..\\data\\...)
        - HF relative path
        - filename only
        """

        # normalize path
        img_path = img_path.replace("\\", "/")

        filename = os.path.basename(img_path)

        # Mapping berdasarkan folder HF 
        if "rlh_ext" in img_path:
            folder = "data/mkn_img/rlh_ext"
        elif "rlh_int" in img_path:
            folder = "data/mkn_img/rlh_int"
        elif "rth_ext" in img_path:
            folder = "data/mkn_img/rth_ext"
        elif "rth_int" in img_path:
            folder = "data/mkn_img/rth_int"
        else:
            raise ValueError(f"Tidak bisa menentukan folder untuk: {img_path}")

        # Build full path
        if self.base_path:
            return os.path.join(self.base_path, folder, filename)
        else:
            return os.path.join(folder, filename)
        

    # Remote Image Loader
    def _load_image_from_url(self, img_path):
        """
        Load image langsung dari Hugging Face (remote)
        """

        img_path = img_path.replace("\\", "/")   # ubah ke slash
        img_path = img_path.replace("../", "")   # hilangkan ../
        img_path = img_path.lstrip("/")          # hilangkan leading /

        if img_path in self.image_cache:
            return self.image_cache[img_path]

        url = f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/{img_path}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")

            self.image_cache[img_path] = img  

            return img

        except Exception as e:
            raise ValueError(f"Gagal load image dari URL: {url} | {e}")
        
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

                    # HF REMOTE
                    if self.dataset_name:
                        img = self._load_image_from_url(img)

                    # LOCAL
                    else:
                        img_path = self._resolve_image_path(img)
                        try:
                            img = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            raise ValueError(f"Gagal load image: {img_path} | {e}")

                # HANDLE PIL IMAGE (HF native)
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