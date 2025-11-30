import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv

from bugulma_enjoyers.constants import LOW_SAFETY
from bugulma_enjoyers.models._api_model import APIModel

logger = logging.getLogger(__name__)


class GoogleModel(APIModel, model_type="google"):
    def __init__(self, model_name: str, pipeline_config: dict) -> None:
        super().__init__(model_name, pipeline_config)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name, safety_settings=LOW_SAFETY)

    def invoke_model(self, input_: str) -> str:
        return self.model.generate_content(prompt=input_).text
