from datasets import load_dataset
from PIL import Image


class VLMdataset:
    def __init__(self, dataset_name=None, data_path=None, split="train"):
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

    def __len__(self):
        return len(self.dataset)

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

                # Handle kalau masih path/string
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")

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