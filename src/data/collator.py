import torch
import logging

logger = logging.getLogger(__name__)


class VLMCollator:
    def __init__(self, device=None):
        """
        Collator untuk menggabungkan multiple formatted samples menjadi batch.

        Args:
            device (str | None):
                - None → biarkan trainer handle (RECOMMENDED)
                - "cuda"/"cpu" → manual device control
        """
        self.device = device

    def __call__(self, batch: list) -> dict:
        """
        Args:
            batch: list of formatted samples dari formatter

        Returns:
            dict: batch tensor siap untuk model
        """

        # 1. VALIDASI INPUT
        required = {"input_ids", "attention_mask", "labels"}
        for i, item in enumerate(batch):
            missing = required - set(item.keys())
            if missing:
                raise ValueError(f"Sample {i} missing fields: {missing}")

        # 2. TEXT TENSORS (STACK)
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        # optional: mm_token_type_ids (Qwen biasanya ada)
        mm_token_type_ids = None
        if "mm_token_type_ids" in batch[0]:
            mm_token_type_ids = torch.stack([
                item["mm_token_type_ids"] for item in batch
            ])

        # 3. IMAGE TENSORS
        pixel_values_list = [item["pixel_values"] for item in batch]

        try:
            pixel_values = torch.stack(pixel_values_list)
        except RuntimeError:
            pixel_values = pixel_values_list
            logger.debug("pixel_values shape beda -> keep as list")

        # 4. IMAGE GRID
        image_grid_list = [item["image_grid_thw"] for item in batch]

        try:
            image_grid_thw = torch.stack(image_grid_list)
        except RuntimeError:
            image_grid_thw = image_grid_list
            logger.debug("image_grid_thw shape beda → keep as list")

        # 5. BUILD BATCH
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if mm_token_type_ids is not None:
            batch_dict["mm_token_type_ids"] = mm_token_type_ids

        batch_dict["pixel_values"] = pixel_values
        batch_dict["image_grid_thw"] = image_grid_thw

        # 6. MOVE TO DEVICE 
        if self.device is not None:
            batch_dict = self._move_to_device(batch_dict)

        return batch_dict

    def _move_to_device(self, batch_dict: dict) -> dict:
        """
        Move tensor (termasuk list tensor) ke device
        """
        result = {}

        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)

            elif isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v):
                result[k] = [x.to(self.device) for x in v]

            else:
                result[k] = v

        return result