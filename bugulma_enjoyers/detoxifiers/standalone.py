"""Class StandaloneDetoxifier: the one that uses single model is defined here."""

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from bugulma_enjoyers.datasets.collate import collate_fn
from bugulma_enjoyers.datasets.detoxification_dataset import DetoxificationDataset
from bugulma_enjoyers.detoxifiers.base import BaseDetoxifier, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class MT0PipelineConfig(PipelineConfig):
    """Configuration for the MT0 detoxifier."""

    detoxifier_model_name: str = "s-nlp/mt0-xl-detox-orpo"


class StandaloneDetoxifier(BaseDetoxifier):
    """The detoxifier consisting of single model (e.g. MT0-XL)."""

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initializes the StandaloneDetoxifier.

        Args:
            config (PipelineConfig): Configuration for the detoxifier

        Notes:
            - The detoxifier model is loaded from Hugging Face Hub.
            - The model is set to evaluation mode after loading.

        """
        self.config = config
        self.device = torch.device(config.device)

        logger.info("Loading MT0-XL model: %s", config.detoxifier_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.detoxifier_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.detoxifier_model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def detoxify(self, text: str, language: str) -> str:
        """
        Runs detoxification on a single text.

        Args:
            text (str): Text to detoxify
            language (str): Language of the text

        Returns:
            str: detoxified text

        """
        results = self.detoxify_batch([text], [language])
        return results[0]

    def detoxify_batch(self, texts: list[str], languages: list[str]) -> list[str]:
        """
        Run detoxification on a collection of texts, batch-by-batch.

        Args:
            texts (List[str]): List of texts to detoxify
            languages (List[str]): List of languages of the texts

        Returns:
            List[str]: List of detoxified texts

        """
        results = []

        dataset = DetoxificationDataset(
            texts=texts,
            languages=languages,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        with torch.inference_mode():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    early_stopping=True,
                )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(decoded)

        return results
