"""Toxic words removal detoxifier based on vocabulary and regex patterns."""

import logging
import re
from typing import Dict, List, Any, Never

import torch
from datasets import load_dataset

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



class ToxicLexicon:
    """Manages toxic word dictionaries and regex patterns for different languages."""

    def __init__(self):
        """Initialize the toxic lexicon with language-specific dictionaries."""
        self.lexicons = self._load_lexicons()
        self.compiled_patterns = self._compile_patterns()
        self.replacement_dict = self._load_replacements()

    def _load_lexicons(self) -> Dict[str, set]:
        """Load toxic word dictionaries for each language from HuggingFace dataset."""
        logger.info("Loading toxic lexicon from textdetox/multilingual_toxic_lexicon dataset")
        try:
            toxic_dataset = load_dataset("textdetox/multilingual_toxic_lexicon")
            lexicons = {}

            # Extract toxic words for each language
            for lang_code in ["en", "ru", "tt"]:
                if lang_code in toxic_dataset:
                    toxic_words = set()
                    for item in toxic_dataset[lang_code]:
                        word = item.get("text", "").strip()
                        if word:
                            toxic_words.add(word.lower())
                    lexicons[lang_code] = toxic_words
                    logger.info(f"Loaded {len(toxic_words)} toxic words for language: {lang_code}")
                else:
                    logger.warning(f"Language {lang_code} not found in dataset")
                    lexicons[lang_code] = set()

            return lexicons
        except Exception as e:
            logger.error(f"Error loading toxic lexicon dataset: {e}")
            logger.warning("Falling back to empty lexicons")
            return {"en": set(), "ru": set(), "tt": set()}

    def _load_replacements(self) -> Dict[str, Dict[str, str]]:
        """Load replacement dictionary for toxic words (fallback if replacement is preferred over removal)."""
        return {
            "en": {
                "fuck": "heck",
                "shit": "stuff",
                "damn": "dang",
            },
            "ru": {
                "блядь": "ну",
                "дебил": "человек",
            },
            "tt": {}  # Tatar replacements can be added as needed
        }

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficient word matching."""
        patterns = {}
        for lang, words in self.lexicons.items():
            if words:
                # Create pattern with word boundaries for accurate matching
                pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
                patterns[lang] = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        return patterns

    def contains_toxic(self, text: str, language: str) -> bool:
        """Check if text contains toxic words."""
        if language not in self.compiled_patterns:
            return False
        return bool(self.compiled_patterns[language].search(text))

    def get_toxic_words(self, text: str, language: str) -> List[str]:
        """Get list of toxic words found in text."""
        if language not in self.compiled_patterns:
            return []
        matches = self.compiled_patterns[language].findall(text)
        return list(set(matches))

    def mask_toxic_words(self, text: str, language: str, mask: str = "") -> str:
        """Mask toxic words in text (remove or replace with mask)."""
        if language not in self.compiled_patterns:
            return text
        return self.compiled_patterns[language].sub(mask, text)

    def replace_toxic_words(self, text: str, language: str) -> str:
        """Replace toxic words with alternatives."""
        if language not in self.replacement_dict:
            return text

        replacements = self.replacement_dict[language]
        for toxic, replacement in replacements.items():
            pattern = rf'\b{re.escape(toxic)}\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.UNICODE)

        return text

    def cleanup_whitespace(self, text: str) -> str:
        """Clean up multiple spaces and strip."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class VocabDetoxifier(BaseModel, model_type="vocab"):
    """
    Detoxifier based on toxic vocabulary removal and replacement.
    
    Uses regex patterns to identify and remove/replace toxic words
    while preserving the overall structure and meaning of the text.
    Compatible with the StandaloneDetoxifier pipeline.
    """

    def __init__(self, model_name: str, pipeline_config: dict, **kwargs: dict) -> None:
        """
        Initialize VocabDetoxifier.
        
        Args:
            model_name: Model name (not used for vocab-based approach)
            pipeline_config: Pipeline configuration
            **kwargs: Additional arguments
        """
        self.lexicon = ToxicLexicon()
        self.config = pipeline_config
        # Provide a dummy tokenizer for compatibility with StandaloneDetoxifier
        self.tokenizer = DummyTokenizer()
        self.device = "cpu"  # Vocab-based approach doesn't use GPU
        logger.info("VocabDetoxifier initialized")

    def _detoxify_text(self, text: str, language: str) -> str:
        """Detoxify a single text using vocabulary-based approach (removal)."""
        if not text or not isinstance(text, str):
            return text

        # Remove toxic words (mask with empty string)
        detoxified = self.lexicon.mask_toxic_words(text, language, mask="")

        # Clean up whitespace
        detoxified = self.lexicon.cleanup_whitespace(detoxified)

        return detoxified

    def forward(self, batch: dict) -> List[str]:
        """
        Process a batch and return detoxified texts.

        Args:
            batch (dict): Batch dictionary containing:
                - original_text: List[str] - texts to detoxify
                - languages: List[str] - language codes

        Returns:
            List[str]: List of detoxified texts
        """
        original_texts = batch.get("original_text", [])
        languages = batch.get("languages", [])

        results = []
        for text, language in zip(original_texts, languages):
            detoxified = self._detoxify_text(text, language)
            results.append(detoxified)

        return results

    def to(self, device: str) -> "VocabDetoxifier":
        """
        Move model to device (no-op for vocab-based detoxifier).

        Args:
            device (str): Device name

        Returns:
            VocabDetoxifier: Self for chaining
        """
        # Vocab-based approach doesn't need to be moved to device
        self.device = device
        return self
