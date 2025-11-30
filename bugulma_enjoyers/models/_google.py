import json
import logging
import os
import re

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

    def forward(self, batch: dict) -> list[str]:
        prompt = batch["prompted_text"]
        try:
            response = self.model.generate_content(prompt)
            text_response = response.text
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
                for key, value in predictions.items():
                    if isinstance(value, list):
                        predictions = value
                        break

            res = batch["original_text"].copy()
            for item in predictions:
                try:
                    idx = item["ID"]
                    detoxified_text = item["tat_detox1"]
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

    def clean_json_response(self, response_text):
        """Очищает ответ от Markdown-тегов для получения чистого JSON."""
        cleaned = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()
