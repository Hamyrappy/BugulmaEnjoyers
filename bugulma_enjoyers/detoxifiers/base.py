"""ABC for detoxifier & basic pipeline config is defined here."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


class BaseDetoxifier(ABC):
    """ABC for detoxifiers."""

    @abstractmethod
    def detoxify(self, text: str, language: str) -> str:
        """
        Detoxify text.

        Args:
            text (str): Text to detoxify
            language (str): Language of the text

        Returns:
            str: detoxified text

        """

    @abstractmethod
    def detoxify_batch(self, texts: list[str], languages: list[str]) -> list[str]:
        """
        Detoxify a batch of texts.

        Args:
            texts (List[str]): List of texts to detoxify
            languages (List[str]): List of languages of the texts

        Returns:
            List[str]: List of detoxified texts

        """


@dataclass
class PipelineConfig:
    """
    MT0 pipeline config.

    Attributes:
        mt0_model_name (str): The name of the MT0 model to use to detoxify.
        nllb_model_name (str): The name of the NLLB model to use for translation.
        toxicity_model_name (str): The name of the model used to detect toxicity.

    """

    # Модели
    detoxifier_model_name: str = "s-nlp/mt0-xl-detox-orpo"
    translator_model_name: str = "facebook/nllb-200-distilled-600M"
    toxicity_detector_model_name: str = "textdetox/xlmr-large-toxicity-classifier-v2"

    # Inference параметры
    max_length: int = 256
    batch_size: int = 8
    num_beams: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False

    toxicity_threshold: float = 0.5
    similarity_threshold: float = 0.7

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    backtranslation_languages: list[str] = field(default_factory=lambda: ["ru"])

    pivot_language: str = "en"


labse_model_name: str = "sentence-transformers/LaBSE"
xcomet_model_name: str = "myyycroft/XCOMET-lite"
