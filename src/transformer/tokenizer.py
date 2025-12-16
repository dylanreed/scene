"""Simple character/word tokenizer for text conditioning."""

import re
from typing import List, Optional


class SimpleTokenizer:
    """Simple word-level tokenizer with hash-based vocabulary.

    Uses hashing to map words to token IDs without needing
    a pre-built vocabulary. Good enough for small-scale training.
    """

    def __init__(self, vocab_size: int = 1000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.pad_token = 0
        self.unk_token = 1
        self.start_token = 2
        self.end_token = 3
        self.special_tokens = 4  # Reserve first 4 IDs

    def _hash_word(self, word: str) -> int:
        """Hash a word to a token ID."""
        # Simple hash function
        h = hash(word.lower()) % (self.vocab_size - self.special_tokens)
        return h + self.special_tokens

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words."""
        # Simple word tokenization
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        return words

    def encode(self, text: str, pad: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string
            pad: Whether to pad/truncate to max_length

        Returns:
            List of token IDs
        """
        words = self._tokenize(text)
        tokens = [self.start_token]
        tokens.extend(self._hash_word(w) for w in words)
        tokens.append(self.end_token)

        if pad:
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length-1] + [self.end_token]
            else:
                tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text (lossy - just for debugging)."""
        # Can't truly decode hash-based tokens, return placeholder
        return f"<{len(tokens)} tokens>"
