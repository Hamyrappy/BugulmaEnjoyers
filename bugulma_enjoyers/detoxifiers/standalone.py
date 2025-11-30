"""Class StandaloneDetoxifier: the one that uses single model is defined here."""

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from bugulma_enjoyers.datasets.collate import get_collate_fn
from bugulma_enjoyers.datasets.detoxification_dataset import DetoxificationDataset
from bugulma_enjoyers.detoxifiers.base import BaseDetoxifier, PipelineConfig
from bugulma_enjoyers.load_model import load_model

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

        """
        self.config = config
        self.device = torch.device(config.device)

        logger.info("Loading Model: %s", config.detoxifier_model_name)
        self.model = load_model(
            model_name=config.detoxifier_model_name,
            pipeline_config=config
        )
        self.model.to(self.device)

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
        dataset = DetoxificationDataset(
            texts=texts,
            languages=languages,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_length,
            use_prompts=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=get_collate_fn(task="detoxification"),
            shuffle=False,
        )

        with torch.inference_mode():
            return [output for batch in dataloader for output in self.model.forward(batch)]
