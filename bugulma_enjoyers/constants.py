"""Useful constants are defined here."""

from enum import Enum


class Language(Enum):
    """Two-letter language codes."""

    ENGLISH = "en"
    RUSSIAN = "ru"
    TATAR = "tt"


NLLB_LANG_CODES: dict[str, str] = {"en": "eng_Latn", "ru": "rus_Cyrl", "tt": "tat_Cyrl"}
