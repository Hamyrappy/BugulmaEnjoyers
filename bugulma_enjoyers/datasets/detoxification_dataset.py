"""Basic detoxification dataset class is defined here."""

import logging

import torch
from torch.utils.data import Dataset

from bugulma_enjoyers.prompts import SIMPLE_PROMPTS

logger = logging.getLogger(__name__)


class DetoxificationDataset(Dataset):
    """Basic detoxification dataset class."""

    def __init__(
        self,
        texts: list[str],
        languages: list[str],
        tokenizer: torch.nn.Module,
        max_length: int = 256,
        task="detoxification",
        use_prompts: bool = True,
        forced_bos_token_id: int | None = None,
    ) -> None:
        """
        Initializes the DetoxificationDataset.

        Args:
            texts (list[str]): List of texts to detoxify.
            languages (list[str]): List of languages of the texts.
            tokenizer (torch.nn.Module): The tokenizer to use for tokenization.
            max_length (int, optional): The maximum length of the input text. Defaults to 256.
            prompts (dict[str, str] | None, optional): The prompts to use for detoxification. Defaults to None.

        """
        self.texts = texts
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forced_bos_token_id = forced_bos_token_id
        self.task = task
        self.use_prompts = use_prompts

    def __len__(self) -> int:
        """The length of the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns the item at the given index.

        The element is a dictionary with the following keys:
            - input_ids: torch.Tensor of shape (max_length,)
            - attention_mask: torch.Tensor of shape (max_length,)
            - language: str
            - original_text: str
        Note that the input_ids and attention_mask correspond to the text with the prompt prepended.
        """
        text = self.texts[idx]
        lang = self.languages[idx]

        prompt = SIMPLE_PROMPTS[self.task][lang] if self.use_prompts else "{}"

        input_text = prompt.format(text)

        # Токенизация
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "language": lang,
            "original_text": text,
            "prompted_text": input_text,
            "forced_bos_token_id": self.forced_bos_token_id,
        }
