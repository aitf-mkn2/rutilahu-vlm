from __future__ import annotations

from transformers import AutoProcessor
import torch
from PIL import Image

from copy import deepcopy
from typing import Any, Dict, List, Optional


class MultimodalPreprocessor:
    def __init__(
        self,
        model_name: str,
        max_length: int = 4096,
        add_generation_prompt: bool = False,
        trust_remote_code: bool = True,
        debug_decode: bool = False,
        return_formatted_text: bool = False,
        ):

        # Load processor (tokenizer + image processor) sesuai model Qwen
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        # Batas maksimum panjang token input
        self.max_length = max_length

        self.add_generation_prompt = add_generation_prompt
        self.debug_decode = debug_decode
        self.return_formatted_text = return_formatted_text
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        
   
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point utama.

        Expected input dari dataset.py -> messages

        Expected output:
        "input_ids": Tensor -> text
        "attention_mask": Tensor -> menandakan token yang valid
        "pixel_values": Tensor -> image
        """

        messages = self._extract_messages(sample)
        formatted_text = self._apply_chat_template(messages)
        images = self._extract_images(messages)

        model_inputs = self._processor_forward(
            formatted_text=formatted_text,
            images=images,
        )

        output = self._squeeze_batch_dimension(model_inputs)

        if self.debug_decode:
            output["decoded_text"] = self._safe_decode(output.get("input_ids"))

        if self.return_formatted_text:
            output["formatted_text"] = formatted_text

        return output

    def _extract_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ambil messages dari sample.
        """
        if not isinstance(sample, dict):
            raise ValueError("Sample harus berupa dict.")

        if "messages" not in sample:
            raise ValueError("Sample tidak punya field `messages`.")

        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Field `messages` harus berupa list non-kosong.")

        return messages
 
    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        """
        Ubah messages menjadi string conversation sesuai template model.
        """
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )
    
    def _extract_images(self, messages: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Ambil semua image dari messages secara berurutan saat chat template dibangun.
        """
        images: List[Image.Image] = []

        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") != "image":
                    continue

                image = item.get("image")
                if image is None:
                    raise ValueError("Ditemukan item image tanpa field `image`.")

                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"Expected PIL.Image.Image, tetapi mendapat {type(image)}. "
                        "Pastikan dataset.py sudah mengonversi path menjadi PIL.Image."
                    )

                images.append(image)

        return images
    
    def _processor_forward(
        self,
        formatted_text: str,
        images: List[Image.Image],
    ) -> Dict[str, Any]:
        """
        Jalankan processor multimodal untuk text + image.
        """
        processor_kwargs: Dict[str, Any] = {
            "text": formatted_text,
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_length,
        }

        if images:
            processor_kwargs["images"] = images
        else:
            processor_kwargs["images"] = None

        model_inputs = self.processor(**processor_kwargs)
        return dict(model_inputs)


    def _squeeze_batch_dimension(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hapus batch dimension karena preprocessor ini memproses per-sample.

        Contoh:
        - [1, seq_len] -> [seq_len]
        - [1, 3, H, W] -> [3, H, W]
        """
        squeezed: Dict[str, Any] = {}

        for key, value in model_inputs.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
                squeezed[key] = value.squeeze(0)
            else:
                squeezed[key] = value

        return squeezed

    def _safe_decode(self, input_ids: Any) -> Optional[str]:
        """
        Debug helper untuk decode input_ids.
        """
        if self.tokenizer is None:
            return None

        if input_ids is None:
            return None

        if not torch.is_tensor(input_ids):
            return None

        try:
            ids = input_ids.tolist()
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return None

    def inspect(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper untuk inspeksi debugging.
        Mengembalikan formatted text, jumlah image, dan model inputs.
        """
        messages = self._extract_messages(sample)
        formatted_text = self._apply_chat_template(messages)
        images = self._extract_images(messages)
        model_inputs = self._squeeze_batch_dimension(
            self._processor_forward(formatted_text=formatted_text, images=images)
        )

        return {
            "formatted_text": formatted_text,
            "num_images": len(images),
            "model_inputs": model_inputs,
            "decoded_text": self._safe_decode(model_inputs.get("input_ids")),
        }