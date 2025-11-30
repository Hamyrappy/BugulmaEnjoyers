"""Useful constants are defined here."""

from enum import Enum

from google.generativeai.types import HarmBlockThreshold, HarmCategory


class Language(Enum):
    """Two-letter language codes."""

    ENGLISH = "en"
    RUSSIAN = "ru"
    TATAR = "tt"


NLLB_LANG_CODES: dict[str, str] = {"en": "eng_Latn", "ru": "rus_Cyrl", "tt": "tat_Cyrl"}


LOW_SAFETY = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}
