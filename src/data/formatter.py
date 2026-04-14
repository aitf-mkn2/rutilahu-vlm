from transformers import AutoProcessor
import torch


class VLMFormatter:
    def __init__(self, model_name: str, max_length: int = 2048):
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
        model_inputs, input_ids = self._tokenize_full(full_text, images)

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
    # Bangun struktur chat (user → assistant) dengan placeholder image + instruction
    def _build_conversation(self, images, instruction, output):
        return [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": instruction},
                ]
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
    def _tokenize_full(self, full_text, images):
        model_inputs = self.processor(
            text=full_text,
            images=images if images else None,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = model_inputs["input_ids"].squeeze(0)
        return model_inputs, input_ids

    # Tokenize user-only text untuk menghitung panjang input user (boundary masking)
    def _tokenize_user(self, user_text):
        return self.processor.tokenizer(
            user_text,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

    # 5. Label Masking
    def _create_labels(self, input_ids, user_ids):
        # Salin input_ids sebagai dasar labels
        labels = input_ids.clone()

        # Tentukan panjang bagian user
        user_len = len(user_ids)

        # Mask bagian user (agar tidak dihitung dalam loss)
        labels[:user_len] = -100

        # Mask padding (agar model tidak belajar dari token kosong)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        return labels

    # 6. Final Output
    def _build_output(self, model_inputs, labels):
        # Hilangkan dimensi batch (karena processing per sample)
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # Tambahkan labels ke dalam input model
        model_inputs["labels"] = labels

        return model_inputs