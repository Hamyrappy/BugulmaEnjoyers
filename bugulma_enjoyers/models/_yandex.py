import json
import os

import requests
from dotenv import load_dotenv

from bugulma_enjoyers.models._api_model import APIModel

NON_45_HTTP_ERROR = "Error: {}, response text: {}"


def ask_yandex_utf8(
    user_text,
    api_key,
    cloud_folder,
    model_name,
    system_role="Ты — умный ассистент.",
    timeout=60,
):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": cloud_folder,
    }

    prompt = {
        "modelUri": f"gpt://{cloud_folder}/{model_name}",
        "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": 1000},
        "messages": [{"role": "system", "text": system_role}, {"role": "user", "text": user_text}],
    }

    # Кодируем в UTF-8 вручную, чтобы не было ошибок с кириллицей
    data_payload = json.dumps(prompt, ensure_ascii=False).encode("utf-8")

    response = requests.post(url, headers=headers, data=data_payload, timeout=timeout)

    response.raise_for_status()

    if response.ok:
        result = response.json()
        return result["result"]["alternatives"][0]["message"]["text"]
    raise requests.exceptions.HTTPError(
        NON_45_HTTP_ERROR.format(response.status_code, response.text),
    )


class YandexModel(APIModel, model_type="yandex"):
    def __init__(self, model_name: str, pipeline_config: dict, **kwargs: dict) -> None:
        super().__init__(model_name, pipeline_config)
        load_dotenv()
        self.model_name = model_name
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.cloud_folder = os.getenv("YANDEX_CLOUD_FOLDER")

    def invoke_model(self, input_: str) -> str:
        return ask_yandex_utf8(
            user_text=input_,
            api_key=self.api_key,
            cloud_folder=self.cloud_folder,
            model_name=self.model_name,
        )
