import json
import logging
import re
from typing import Any, Never

import torch

from bugulma_enjoyers.models._base import BaseModel

logger = logging.getLogger(__name__)


class DummyTokenizer:
    """
    Dummy tokenizer.

    A dummy tokenizer class used to satisfy the interface requirements of the detoxification pipeline.
    It does not perform any actual tokenization and always returns empty tensors.
    """

    def encode(self, *args: Any, **kwds: Any) -> torch.Tensor:
        """
        Encode a single text into a tensor. Returns an empty tensor.

        Args:
            text (str): The text to encode.

        Returns:
            torch.Tensor: The encoded tensor.

        """
        return {"input_ids": torch.tensor((0,)), "attention_mask": torch.tensor((0,))}

    def encode_batch(self, *args: Any, **kwds: Any) -> torch.Tensor:
        """
        Encode a batch of texts into tensors. Returns an empty tensor.

        Args:
            texts (List[str]): The list of texts to encode.

        Returns:
            torch.Tensor: The encoded tensor.

        """
        return {"input_ids": torch.tensor((0, 0)), "attention_mask": torch.tensor((0, 0))}

    def decode(self, token_ids: torch.Tensor) -> Never:
        """
        Decode a single tensor into a text. Raises NotImplementedError.

        Args:
            token_ids (torch.Tensor): The tensor to decode.

        Raises:
            NotImplementedError: Decoding is not supported for the dummy tokenizer.

        """

    def decode_batch(self, batch_token_ids: torch.Tensor) -> Never:
        """
        Decode a batch of tensors into texts. Raises NotImplementedError.

        Args:
            batch_token_ids (torch.Tensor): The batch of tensors to decode.

        Raises:
            NotImplementedError: Decoding is not supported for the dummy tokenizer.

        """

    def to(self, device: torch.device) -> None:
        """
        Move the tokenizer to the specified device. Does nothing.

        Args:
            device (torch.device): The device to move the tokenizer to.

        """

    def convert_tokens_to_ids(self, token: str) -> int:
        """
        Convert a token to its corresponding ID. Raises NotImplementedError.

        Args:
            token (str): The token to convert.

        Raises:
            NotImplementedError: Conversion is not supported for the dummy tokenizer.

        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.encode(*args, **kwds)


class APIModel(BaseModel, model_type="api"):
    """Base model class for API-based models."""

    def __init__(self, model_name: str, pipeline_config: dict, **kwargs: dict) -> None:
        """Initializes the APIModel."""
        self.tokenizer = DummyTokenizer()

    def invoke_model(self, input_: str) -> str:
        """Invokes the model on the input text."""
        raise NotImplementedError

    def forward(self, batch: dict) -> list[str]:
        prompt = batch["prompted_text"]
        try:
            text_response = self.invoke_model(prompt)
            # Логика извлечения JSON (чуть упрощена для надежности)
            if "```" in text_response:
                # Пытаемся найти блок кода, если он есть
                match = re.search(r"```(?:json)?(.*?)```", text_response, re.DOTALL)
                if match:
                    text_response = match.group(1)

            json_str = self.clean_json_response(text_response)

            # Парсим JSON
            predictions = json.loads(json_str)

            # Если модель вернула объект с ключом (например {"result": [...]}), извлекаем список
            if isinstance(predictions, dict):
                # Ищем любой ключ, который содержит список
                for value in predictions.values():
                    if isinstance(value, list):
                        predictions = value
                        break

            res = batch["original_text"].copy()
            for item in predictions:
                try:
                    idx = item["ID"]
                    if "tat_detox1" in item:
                        detoxified_text = item["tat_detox1"]
                    elif "detoxified_text" in item:
                        detoxified_text = item["detoxified_text"]
                    else:
                        detoxified_text = item["text"]
                    res[idx] = detoxified_text
                except KeyError:
                    logger.warning("Missing expected keys in model output item: %s", item)

        except Exception:
            logger.exception("Error parsing model response")
            # Возвращаем пустой список, чтобы пайплайн не падал,
            # или можно вернуть оригинальные тексты как 'fail-safe'
            return batch["original_text"]

        else:
            return res

    def to(self, device: str) -> None:
        """Moves the model to the specified device. Does nothing for API-based models."""

    def clean_json_response(self, response_text) -> str:
        """Очищает ответ от Markdown-тегов для получения чистого JSON."""
        cleaned = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()
