from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .image_cache import ImageCache
from .image_loader import ImageResolver
from .loaders import load_dataset_split
from .normalizers import normalize_messages


class VLMdataset:
    """
    Tugas class ini:
    - load dataset mentah
    - siapkan resolver image
    - siapkan cache
    - normalize sample saat __getitem__
    """
    def __init__(
        self, 
        dataset_name: Optional[str] = None,
        data_path: Optional[Union[str, Path, Dict[str, str]]] = None,
        split: str = "train",
        base_path: str = "",
        image_root: Optional[str] = None,
        cache_images: bool = True,
        cache_size: int = 128,
        verify_images: bool = True,
        strict_validation: bool = True,
        allow_hf_fallback: bool = True,
        require_system: bool = True,
        allow_url_loading: bool = False,
        hf_split_files: Optional[Dict[str, str]] = None,
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.split = split
        self.base_path = base_path
        self.image_root = (
            Path(image_root or base_path).expanduser() 
            if (image_root or base_path) 
            else None
        )

        self.cache_images = cache_images
        self.cache_size = max(int(cache_size), 0)
        self.verify_images = verify_images
        self.strict_validation = strict_validation
        self.allow_hf_fallback = allow_hf_fallback
        self.require_system = require_system
        self.allow_url_loading = allow_url_loading
        self.hf_split_files = hf_split_files or {}

        self._image_cache: "OrderedDict[tuple[str, str], Image.Image]" = OrderedDict()

        self.dataset = self._load_dataset()

        self.image_resolver = ImageResolver(
            image_root=self.image_root,
            dataset_name=self.dataset_name,
            allow_hf_fallback=self.allow_hf_fallback,
            allow_url_loading=self.allow_url_loading,
            verify_images=self.verify_images,
            cache=self.image_cache,
            hf_split_files=self.hf_split_files,
        )

        self.dataset = load_dataset_split(
            data_path=self.data_path,
            dataset_name=self.dataset_name,
            split=self.split,
            allow_hf_fallback=self.allow_hf_fallback,
            hf_split_files=self.hf_split_files,
        )
         
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        """
        Ambil satu sample dan kembalikan dalam format conversation yang sudah dinormalisasi.

        Output utama:
        {
            "messages": [
                {"role": "system", "content": [...]},
                {"role": "user", "content": [...]},
                {"role": "assistant", "content": [...]}
            ]
        }
        """
        raw_sample = self.dataset[idx]
        sample = deepcopy(raw_sample)

        if "messages" not in sample:
            raise ValueError(f"Sample index {idx} tidak punya field `messages`.")

        messages = sample["messages"]

        normalized_messages = normalize_messages(
            messages=messages,
            idx=idx,
            require_system=self.require_system,
            strict_validation=self.strict_validation,
            image_resolver=self.image_resolver,
        )

        result: Dict[str, Any] = {"messages": normalized_messages}

        # Simpan metadata debug kalau ada
        for key in ("id", "sample_id", "source_id", "source_group_id"):
            if key in sample:
                result[key] = sample[key]

        return result