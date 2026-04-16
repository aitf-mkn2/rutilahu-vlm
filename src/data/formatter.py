from transformers import AutoProcessor
import torch


class VLMFormatter:
    def __init__(self, model_name: str, max_length: int = 4096):
        # Load processor (tokenizer + image processor) sesuai model Qwen
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        # Batas maksimum panjang token input
        self.max_length = max_length

    # Entry Point
    def __call__(self, sample: dict) -> dict:
        # Ekstrak komponen utama dari sample (image, instruction, output)
        images, instruction, output = self._extract_sample(sample)

        # Susun conversation (user + assistant) sesuai format chat model
        conversation = self._build_conversation(images, instruction, output)

        # Convert conversation → formatted text (training: user + assistant)
        full_text = self._apply_template(conversation, is_training=True)

        # Generate user-only text (dengan prompt Assistant:) untuk menentukan boundary masking
        user_text = self._apply_template([conversation[0]], is_training=False)

        # Tokenize full input (text + image) menjadi tensor untuk model
        model_inputs = self._tokenize(full_text, images)
        input_ids = model_inputs["input_ids"].squeeze(0)

        # Tokenize user-only text untuk mengetahui panjang input user
        user_ids = self._tokenize_user(user_text)

        # Buat label dan lakukan masking (agar model hanya belajar dari output)
        labels = self._create_labels(input_ids, user_ids)

        # Gabungkan semua input + labels menjadi format final untuk training
        return self._build_output(model_inputs, labels)

    # 1. Extract
    # Ambil dan bersihkan field utama dari dataset
    def _extract_sample(self, sample):
        return (
            sample["images"],
            sample["instruction"].strip(),
            sample["output"].strip()
        )

    # 2. Conversation
    # Bangun struktur chat (user → assistant) dengan placeholder instruction
    def _build_conversation(self, images, instruction, output):
        user_content = []

        for _ in images:
            user_content.append({"type": "image"})  

        user_content.append({"type": "text", "text": instruction})

        return [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": output}
                ]
            }
        ]

    # 3. Template
    # Ubah conversation menjadi string sesuai template chat Qwen (dengan special tokens)
    def _apply_template(self, conversation, is_training=True):
        return self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=not is_training
        )

    # 4. Tokenization
    # Convert text + image menjadi tensor (input_ids, attention_mask, pixel_values)
    def _tokenize(self, full_text, images):
        model_inputs = self.processor(
            text=full_text,
            images=images if images else None,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length
        )
        return model_inputs


    def _tokenize_user(self, user_text):
        return self.processor.tokenizer(
            user_text,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

    # 5. Label Masking
    def _create_labels(self, input_ids, user_ids):
        labels = input_ids.clone()
        tokenizer = self.processor.tokenizer

        assistant_candidates = [
            "<|im_start|>assistant",  
            "<|assistant|>",          
            "assistant"
        ]

        start_idx = None

        for candidate in assistant_candidates:
            token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]

            for i in range(len(input_ids) - len(token_ids) + 1):
                if input_ids[i:i+len(token_ids)].tolist() == token_ids:
                    start_idx = i + len(token_ids)
                    print(f"[DEBUG] Found assistant token: {candidate}")
                    break

            if start_idx is not None:
                break

        if start_idx is None:
            print("WARNING! Assistant token tidak ditemukan!")
            print("WARNING! Using fallback user_ids (might be inaccurate)")           
            start_idx = min(len(user_ids), len(input_ids) - 1)

        labels[:start_idx] = -100

        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        unmasked = (labels != -100).sum().item()
        if unmasked == 0:
            print("ERROR! Semua label ter-mask! Model tidak akan belajar!")
        
        return labels

    # 6. Final Output
    def _build_output(self, model_inputs, labels):
        # Hilangkan dimensi batch (karena processing per sample)
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # Tambahkan labels ke dalam input model
        model_inputs["labels"] = labels

        return model_inputs