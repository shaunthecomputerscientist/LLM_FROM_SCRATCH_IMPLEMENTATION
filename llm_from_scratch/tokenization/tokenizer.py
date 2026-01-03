"""
Tokenizer implementation for LLM from scratch.

This module contains the SimpleTokenizerV2 class which handles:
- Text tokenization with special tokens (<|unk|>, 
"""

import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        # 1. Store the word-to-ID dictionary
        self.str_to_int = vocab
        
        # 2. Create the reverse ID-to-word dictionary for decoding
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """Converts text (string) into a list of Token IDs (integers)."""
        # Split text into words and punctuation, keeping punctuation as separate tokens
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        # Remove whitespace and empty strings
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        
        # Replace any word NOT in vocab with the <|unk|> (unknown) token ID
        ids = [
            self.str_to_int.get(s, self.str_to_int["<|unk|>"]) 
            for s in preprocessed
        ]
        return ids

    def decode(self, ids):
        """Converts Token IDs (integers) back into human-readable text."""
        # Join the words back together with spaces
        text = " ".join([self.int_to_str[i] for i in ids])
        
        # Use regex to remove spaces before punctuation (e.g., "Hello !" -> "Hello!")
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text