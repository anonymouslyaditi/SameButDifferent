"""Text preprocessing utilities for paraphrase generation."""

import re
from typing import List, Optional


class TextPreprocessor:
    """Handles text preprocessing for paraphrase generation."""
    
    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 400,
        remove_special_chars: bool = False
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.remove_special_chars = remove_special_chars
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optionally remove special characters
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def is_valid_length(self, text: str) -> bool:
        """Check if text length is within bounds."""
        word_count = self.count_words(text)
        return self.min_words <= word_count <= self.max_words
    
    def truncate_text(self, text: str, max_words: Optional[int] = None) -> str:
        """Truncate text to maximum word count."""
        max_words = max_words or self.max_words
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
        return ' '.join(words)
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs for batch processing."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def prepare_for_model(self, text: str, prefix: str = "paraphrase: ") -> str:
        """Prepare text for model input."""
        text = self.clean_text(text)
        if self.max_words:
            text = self.truncate_text(text)
        return f"{prefix}{text}"
    
    def calculate_output_length_constraint(self, input_text: str, min_ratio: float = 0.8) -> int:
        """Calculate minimum output length based on input."""
        input_word_count = self.count_words(input_text)
        return int(input_word_count * min_ratio)

