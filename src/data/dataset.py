from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.data.image_cache import ImageCache
from src.data.dataset_loader import load_dataset_split
from src.data.image_loader import ImageResolver

logger = logging.getLogger(__name__)

class MultimodalChatDataset:
    """
    Tugas class ini:
    - load raw split dari dataset_loader.py
    - preserve full messages dan urutan multimodal
    - replace image path -> PIL.Image via ImageResolver
    - preserve metadata penting
    - return sample siap masuk preprocessor.py
    """

    VALID_ROLES = {"system", "user", "assistant"}
    METADATA_KEYS = ("id",)

    def __init__(
        self, 
        data_path: Optional[Union[str, Path, Dict[str, str]]] = None,
        split: str = "train",
        image_root: Optional[Union[str, Path]] = None,
        cache_images: bool = True,
        cache_size: int = 128,
        verify_images: bool = True,
        strict_validation: bool = True,
        debug_mode: bool = False,
    ):
        self.data_path = data_path
        self.split = split
        self.image_root = Path(image_root).expanduser() if image_root else None

        self.cache_images = cache_images
        self.cache_size = max(int(cache_size), 0)
        self.verify_images = verify_images
        self.strict_validation = strict_validation
        self.debug_mode = debug_mode

        self.image_cache = (
            ImageCache(max_size=self.cache_size)
            if self.cache_images and self.cache_size > 0
            else None
        )
        
        self.image_resolver = ImageResolver(
            image_root=self.image_root,
            verify_images=self.verify_images,
            cache=self.image_cache,
        )

        self.dataset = load_dataset_split(
            data_path=self.data_path,
            split=self.split,
        )
         
        if self.debug_mode:
            logger.info(
                "MultimodalChatDataset initialized | split=%s | num_samples=%d | image_root=%s",
                self.split,
                len(self.dataset),
                self.image_root,
            )

    def __len__(self) -> int:
        return len(self.dataset)
    

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Ambil 1 sample raw lalu ubah image path menjadi PIL.Image
        tanpa mengubah urutan multimodal conversation.
        """
        raw_sample = self.dataset[idx]
        sample = deepcopy(raw_sample)

        if "messages" not in sample:
            raise ValueError(f"Sample index {idx} tidak punya field `messages`.")

        messages = sample["messages"]
        processed_messages = self._process_messages(messages, idx=idx)

        result: Dict[str, Any] = {"messages": processed_messages}

        for key in self.METADATA_KEYS:
            if key in sample:
                result[key] = sample[key]

        if self.debug_mode:
            logger.debug(
                "Processed sample idx=%d | metadata=%s",
                idx,
                {k: result[k] for k in self.METADATA_KEYS if k in result},
            )

        return result
    
    def get_raw_sample(self, idx: int) -> Dict[str, Any]:
        """Ambil sample mentah tanpa preprocessing image."""
        return deepcopy(self.dataset[idx])
    
    def preview_sample(self, idx: int = 0) -> Dict[str, Any]:
        """Ambil sample yang sudah diproses."""
        return self[idx]
    
    def _process_messages(self, messages: Any, idx: int) -> List[Dict[str, Any]]:
        """
        Process seluruh messages sambil mempertahankan urutan aslinya.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError(f"Sample index {idx}: `messages` harus list non-kosong.")

        processed_messages: List[Dict[str, Any]] = []

        for message_idx, message in enumerate(messages):
            processed_message = self._process_message(
                message=message,
                idx=idx,
                message_idx=message_idx,
            )
            processed_messages.append(processed_message)

        return processed_messages

    def _process_message(
        self,
        message: Any,
        idx: int,
        message_idx: int,
    ) -> Dict[str, Any]:
        """
        Process satu message:
        - validasi role
        - validasi content
        - replace image path -> PIL.Image
        """
        if not isinstance(message, dict):
            raise ValueError(
                f"Sample index {idx}, message {message_idx}: setiap message harus dict."
            )

        if "role" not in message:
            raise ValueError(
                f"Sample index {idx}, message {message_idx}: field `role` tidak ditemukan."
            )

        if "content" not in message:
            raise ValueError(
                f"Sample index {idx}, message {message_idx}: field `content` tidak ditemukan."
            )

        role = message["role"]
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Sample index {idx}, message {message_idx}: role tidak valid: {role}"
            )

        content = message["content"]
        processed_content = self._process_content(
            content=content,
            role=role,
            idx=idx,
            message_idx=message_idx,
            
        )

        processed_message = deepcopy(message)
        processed_message["content"] = processed_content
        return processed_message
    
    def _process_content(
        self,
        content: Any,
        role: str,
        idx: int,
        message_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Process seluruh content dalam satu message.
        Harus berupa list of dict multimodal item.

        Support:
        - string text sederhana
        - list multimodal item
        """
        
        if isinstance(content, str):
            text = content.strip()

            if self.strict_validation and not text:
                raise ValueError(
                    f"Sample index {idx}, message {message_idx} ({role}): text kosong."
                )

            return [
                {
                    "type": "text",
                    "text": text,
                }
            ]

        # Content multimodal wajib list
        if not isinstance(content, list):
            raise ValueError(
                f"Sample index {idx}, message {message_idx} ({role}): `content` harus berupa list."
            )

        if len(content) == 0:
            raise ValueError(
                f"Sample index {idx}, message {message_idx} ({role}): `content` tidak boleh kosong."
            )

        processed_items: List[Dict[str, Any]] = []

        for content_idx, item in enumerate(content):
            processed_item = self._process_content_item(
                item=item,
                role=role,
                idx=idx,
                message_idx=message_idx,
                content_idx=content_idx,
            )
            processed_items.append(processed_item)

        return processed_items
        
    
    def _process_content_item(
        self,
        item: Any,
        role: str,
        idx: int,
        message_idx: int,
        content_idx: int,
    ) -> Dict[str, Any]:
        """
        Process satu item di dalam content.
        Support untuk:
        - text
        - image
        """
        if not isinstance(item, dict):
            raise ValueError(
                f"Sample index {idx}, message {message_idx}, content {content_idx} ({role}): item harus dict."
            )

        item_type = item.get("type")
        if item_type is None:
            raise ValueError(
                f"Sample index {idx}, message {message_idx}, content {content_idx} ({role}): field `type` tidak ditemukan."
            )

        normalized_item = deepcopy(item)

        if item_type == "text":
            text = normalized_item.get("text", "")
            if not isinstance(text, str):
                text = str(text)

            text = text.strip()
            if self.strict_validation and not text:
                raise ValueError(
                    f"Sample index {idx}, message {message_idx}, content {content_idx} ({role}): text kosong."
                )

            normalized_item["text"] = text
            return normalized_item

        if item_type == "image":
            image_ref = normalized_item.get("image")
            if image_ref is None:
                raise ValueError(
                    f"Sample index {idx}, message {message_idx}, content {content_idx} ({role}): field `image` tidak ditemukan."
                )

            pil_image = self.image_resolver.load(
                image_ref=image_ref,
                idx=idx,
                message_idx=message_idx,
                content_idx=content_idx,
            )
            normalized_item["image"] = pil_image
            return normalized_item

        if self.strict_validation:
            raise ValueError(
                f"Sample index {idx}, message {message_idx}, content {content_idx} ({role}): type tidak dikenali: {item_type}"
            )

        return normalized_item
