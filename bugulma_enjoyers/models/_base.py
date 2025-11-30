"""Contains the BaseModel class, which is an abstract base class for all models."""

from abc import ABC, abstractmethod

MODEL_TYPES = {}


class BaseModel(ABC):
    """
    Base model class.

    This class is an abstract base class for all models. It defines two abstract methods:
        - forward: This method should take a batch of data and return a string representing the model's output.
        - preprocess: This method should take a text and its language and return a preprocessed string.

    The class is abstract and cannot be instantiated directly. It is intended to be used as a base class for other models.
    """

    @abstractmethod
    def __init__(self, model_name: str, pipeline_config: dict, **kwargs: dict) -> None:
        """Initializes the BaseModel."""

    @abstractmethod
    def forward(self, batch: dict) -> str:
        """
        Takes a batch of data and return a string representing the model's output.

        Args:
            batch (dict): A dictionary containing the batch of data.

        Returns:
            str: A string representing the model's output.

        """

    def __init_subclass__(cls, model_type: str):
        super().__init_subclass__()
        MODEL_TYPES[model_type] = cls

    @abstractmethod
    def to(self, device: str) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (str): The device to move the model to.

        """
