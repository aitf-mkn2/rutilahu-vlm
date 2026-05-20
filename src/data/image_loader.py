from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .image_cache import CacheKey, ImageCache

logger = logging.getLogger(__name__)

class ImageResolver:
    """
    Resolver untuk mengubah referensi image menjadi PIL.Image.

    Tugas utama:
    - cek cache dulu
    - resolve path lokal
    - load image lokal
    - validasi image
    - simpan ke cache
    - return final PIL.Image
    """

    def __init__(
        self,
        image_root: Optional[Union[str, Path]] = None,
        verify_images: bool = True,
        cache: Optional[ImageCache] = None,
    ):
        self.image_root = Path(image_root).expanduser() if image_root else None
        self.verify_images = verify_images
        self.cache = cache

    def load(
        self,
        image_ref: Union[str, Path, Image.Image],
        idx: Optional[int] = None,
        message_idx: Optional[int] = None,
        content_idx: Optional[int] = None,
    ) -> Image.Image:
        """
        Entry point utama.

        Input bisa:
        - PIL.Image langsung
        - string path relatif / absolut

        Output selalu PIL.Image yang sudah RGB.
        """

        # 1) PIL Image
        if isinstance(image_ref, Image.Image):
            return self._finalize_image(
                image_ref,
                idx=idx,
                message_idx=message_idx,
                content_idx=content_idx,
            )

        if isinstance(image_ref, Path):
            image_ref = str(image_ref)

        if not isinstance(image_ref, str):
            raise ValueError(
                f"image_ref harus str / Path / PIL.Image, tetapi mendapat {type(image_ref)}"
            )

        image_ref = image_ref.strip()
        if not image_ref:
            raise ValueError("image_ref kosong.")


        # 2) Local Path
        local_path = self._resolve_local_image_path(image_ref)
        if local_path is None:
            raise FileNotFoundError(f"Gagal resolve local image path: {image_ref}")

        cache_key = self._make_cache_key("local", str(local_path.resolve()))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        image = self._load_image_from_local(
            local_path,
            idx=idx,
            message_idx=message_idx,
            content_idx=content_idx,
        )
        self._cache_put(cache_key, image)
        return image
    
    def _make_cache_key(self, source_type: str, identifier: str) -> CacheKey:
        return (source_type, identifier)

    def _cache_get(self, key: CacheKey) -> Optional[Image.Image]:
        """Ambil image dari cache kalau ada."""
        if self.cache is None:
            return None
        return self.cache.get(key)
    
    def _cache_put(self, key: CacheKey, image: Image.Image) -> None:
        """Simpan image ke cache."""
        if self.cache is None:
            return
        self.cache.put(key, image)

    def _normalize_relative_image_path(self, image_ref: str) -> str:
        """
        Normalisasi path image agar konsisten lintas OS.
        """
        path = image_ref.replace("\\", "/").strip()

        while path.startswith("./"):
            path = path[2:]

        while path.startswith("../"):
            path = path[3:]

        return Path(path).as_posix()
    
    def _resolve_local_image_path(self, image_ref: str) -> Optional[Path]:
        """
        Cari path lokal yang paling masuk akal.

        Prioritas:
        1. absolute path yang valid
        2. image_root / normalized_relative_path
        3. image_root / basename
        4. path relatif apa adanya
        """

        normalized = self._normalize_relative_image_path(image_ref)
        raw_path = Path(image_ref)

        if raw_path.is_absolute() and raw_path.exists():
            return raw_path

        candidates = []

        if self.image_root is not None:
            root = self.image_root
            candidates.append(root / normalized)
            candidates.append(root / Path(normalized).name)

        candidates.append(Path(normalized))
        candidates.append(raw_path)

        seen = set()
        for candidate in candidates:
            try:
                candidate = candidate.expanduser()
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)

                if candidate.exists():
                    return candidate
            except Exception:
                continue

        return None
    
    def _load_image_from_local(
        self,
        image_path: Path,
        idx: Optional[int] = None,
        message_idx: Optional[int] = None,
        content_idx: Optional[int] = None,
    ) -> Image.Image:
        """Load image dari file lokal"""
        if self.verify_images and not image_path.exists():
            raise FileNotFoundError(f"Image lokal tidak ditemukan: {image_path}")

        try:
            with Image.open(image_path) as img:
                return self._finalize_image(
                    img,
                    idx=idx,
                    message_idx=message_idx,
                    content_idx=content_idx,
                )
        except Exception as e:
            raise ValueError(f"Gagal load image lokal: {image_path} | {e}") from e
        
    def _finalize_image(
        self,
        img: Image.Image,
        idx: Optional[int] = None,
        message_idx: Optional[int] = None,
        content_idx: Optional[int] = None,
    ) -> Image.Image:
        """
        Convert image ke RGB, force load ke memory, lalu validasi
        """
        rgb = img.convert("RGB")
        rgb.load()
        self._validate_loaded_image(
            rgb,
            idx=idx,
            message_idx=message_idx,
            content_idx=content_idx,
        )
        return rgb
    
    def _validate_loaded_image(
        self,
        img: Image.Image,
        idx: Optional[int] = None,
        message_idx: Optional[int] = None,
        content_idx: Optional[int] = None,
    ) -> None:
        """Validasi dasar image hasil load."""
        if not self.verify_images:
            return

        if img.mode != "RGB":
            raise ValueError(f"Image mode harus RGB, mendapat {img.mode}.")

        if img.size[0] <= 0 or img.size[1] <= 0:
            raise ValueError(f"Ukuran image tidak valid: {img.size}")

        if idx is not None and message_idx is not None and content_idx is not None:
            logger.debug(
                "Loaded valid image for sample=%s, message=%s, content=%s, size=%s",
                idx,
                message_idx,
                content_idx,
                img.size,
            )    
