from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

from PIL import Image


CacheKey = Tuple[str, str]


class ImageCache:
    """
    LRU cache sederhana untuk menyimpan PIL Image di RAM.
    """

    def __init__(self, max_size: int = 128):
        self.max_size = max(int(max_size), 0)
        self._cache: OrderedDict[CacheKey, Image.Image] = OrderedDict()

    def get(self, key: CacheKey) -> Optional[Image.Image]:
        """
        Ambil image dari cache.

        Kalau key ada, item itu dipindahkan ke urutan paling belakang
        agar dianggap paling baru dipakai (LRU refresh).
        """
        if key not in self._cache:
            return None

        image = self._cache.pop(key)
        self._cache[key] = image
        return image

    def put(self, key: CacheKey, image: Image.Image) -> None:
        """
        Simpan image ke cache.

        Kalau cache sudah melebihi max_size,
        item yang paling lama tidak dipakai akan dibuang.
        """
        if self.max_size <= 0:
            return

        if key in self._cache:
            self._cache.pop(key)

        self._cache[key] = image

        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Hapus semua isi cache."""
        self._cache.clear()

    def __contains__(self, key: CacheKey) -> bool:
        """Untuk mengecek apakah image sudah dicache"""
        return key in self._cache

    def __len__(self) -> int:
        """Jumlah item yang sedang tersimpan di cache."""
        return len(self._cache)

    def __repr__(self) -> str:
        return f"ImageCache(max_size={self.max_size}, current_size={len(self._cache)})"