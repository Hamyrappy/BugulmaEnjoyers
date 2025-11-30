from typing import Any, Never

import torch

from bugulma_enjoyers.models._base import BaseModel


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
        return [self.invoke_model(text) for text in batch["prompted_text"]]

    def to(self, device: str) -> None:
        """Moves the model to the specified device. Does nothing for API-based models."""
