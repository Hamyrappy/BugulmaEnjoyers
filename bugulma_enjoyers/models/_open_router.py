import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

from bugulma_enjoyers.models._api_model import APIModel

logger = logging.getLogger(__name__)


class OpenRouterModel(APIModel, model_type="open_router"):
    def __init__(self, model_name: str, pipeline_config: dict) -> None:
        super().__init__(model_name, pipeline_config)
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def invoke_model(self, input_: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_}],
        )
        return completion.choices[0].message.content
