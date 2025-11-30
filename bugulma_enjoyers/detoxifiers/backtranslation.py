import logging
from collections import defaultdict

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from bugulma_enjoyers.constants import NLLB_LANG_CODES
from bugulma_enjoyers.detoxifiers.base import BaseDetoxifier, PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNKNOWN_LANGUAGE_ERROR = "Language {} is not supported. Supported languages: {}"


class BacktranslationDetoxifier(BaseDetoxifier):
    """Детоксификатор с использованием backtranslation через pivot язык."""

    def __init__(self, config: PipelineConfig, base_detoxifier: BaseDetoxifier) -> None:
        self.config = config
        self.base_detoxifier = base_detoxifier
        self.device = torch.device(config.device)

        logger.info("Loading translator model: %s", config.translator_model_name)
        self.translator_tokenizer = AutoTokenizer.from_pretrained(config.translator_model_name)
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.translator_model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        ).to(self.device)
        self.translator_model.eval()

    def _get_translator_code(self, lang: str) -> str:
        code = NLLB_LANG_CODES.get(lang)
        if code is None:
            raise ValueError(UNKNOWN_LANGUAGE_ERROR.format(lang, list(NLLB_LANG_CODES.keys())))
        return code

    def _translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """
        Перевод текста с одного языка на другой.

        Args:
            text (str): Текст для перевода
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык

        Returns:
            str: Переведенный текст

        """
        src_code = self._get_translator_code(src_lang)
        tgt_code = self._get_translator_code(tgt_lang)
        logger.debug("Translating from %s to %s", src_code, tgt_code)

        inputs = self.translator_tokenizer(text, return_tensors="pt").to(self.device)

        translated_tokens = self.translator_model.generate(
            **inputs,
            forced_bos_token_id=self.translator_tokenizer.convert_tokens_to_ids(tgt_code),
        )

        return self.translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    def _translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        src_code = NLLB_LANG_CODES.get(src_lang, "eng_Latn")
        tgt_code = NLLB_LANG_CODES.get(tgt_lang, "eng_Latn")

        self.translator_tokenizer.src_lang = src_code

        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]

            inputs = self.translator_tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            forced_bos_token_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_code)

            with torch.no_grad():
                outputs = self.translator_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                )

            decoded = self.translator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

        return results

    def detoxify(self, text: str, language: str) -> str:
        """
        Run detoxification on a single text.

        Args:
            text (str): Text to detoxify
            language (str): Language of the text

        Returns:
            str: detoxified text

        Steps:
            1. Translate text to pivot language
            2. Detoxify text on pivot language
            3. Translate detoxified text back to original language

        """
        pivot = self.config.pivot_language

        # Шаг 1: Перевод на pivot язык
        translated = self._translate(text, language, pivot)
        logger.debug("Translated: %s", translated)

        # Шаг 2: Детоксификация на pivot языке
        detoxified = self.base_detoxifier.detoxify(translated, pivot)
        logger.debug("Detoxified: %s", detoxified)

        # Шаг 3: Перевод обратно на исходный язык
        result = self._translate(detoxified, pivot, language)
        logger.debug("Result: %s", result)

        return result

    def detoxify_batch(self, texts: list[str], languages: list[str]) -> list[str]:
        """
        Run detoxification on a batch of texts.

        Args:
            texts (List[str]): List of texts to detoxify
            languages (List[str]): List of languages of the texts

        Returns:
            List[str]: List of detoxified texts

        Steps:
            1. Group texts by language
            2. Translate each group to pivot language
            3. Detoxify each group on pivot language
            4. Translate each group back to original language
            5. Save results

        """
        pivot = self.config.pivot_language
        # Группируем по языкам для эффективности
        lang_groups = defaultdict(list)
        lang_indices = defaultdict(list)
        for idx, (text, lang) in enumerate(zip(texts, languages, strict=True)):
            lang_groups[lang].append(text)
            lang_indices[lang].append(idx)
        results = [None] * len(texts)
        for lang, group_texts in lang_groups.items():
            # Перевод на pivot
            translated = self._translate_batch(group_texts, lang, pivot)
            # print(translated) # noqa: ERA001
            # Детоксификация
            detoxified = self.base_detoxifier.detoxify_batch(translated, pivot)
            # Перевод обратно
            back_translated = self._translate_batch(detoxified, pivot, lang)
            # Сохраняем результаты
            for idx, result in zip(lang_indices[lang], back_translated, strict=True):
                results[idx] = result
        return results
