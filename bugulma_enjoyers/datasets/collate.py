"""Collate function для DataLoader."""

import torch


def collate_fn(batch: list[dict]) -> dict:
    """Collate function для DataLoader."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "languages": [item["language"] for item in batch],
        "original_texts": [item["original_text"] for item in batch],
    }
