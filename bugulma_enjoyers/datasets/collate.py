"""Collate function для DataLoader."""

import json

import torch

from bugulma_enjoyers.prompts import BATCH_PROMPTS


def get_collate_fn(task: str) -> None:
    def collate_fn(batch: list[dict]) -> dict:
        """Collate function для DataLoader."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "languages": [item["language"] for item in batch],
            "original_text": [item["original_text"] for item in batch],
            "prompted_text": ""
            if not batch
            else (
                BATCH_PROMPTS[task][batch[0]["language"]].format(
                    batch_data_str=json.dumps(
                        [
                            {"ID": idx, "text": item["original_text"]}
                            for idx, item in enumerate(batch)
                        ],
                        ensure_ascii=False,
                    ),
                )
            ),
            "forced_bos_token_id": batch[0].get("forced_bos_token_id") if batch else None,
        }

    return collate_fn
