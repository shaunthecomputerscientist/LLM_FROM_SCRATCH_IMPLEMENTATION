"""
DEMONSTRATION: Token Count vs Vocabulary Size
Shows actual numbers and the critical difference
"""

import tiktoken

print("=" * 80)
print("TOKEN COUNT vs VOCABULARY SIZE - THE CRITICAL DIFFERENCE")
print("=" * 80)

# Initialize GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

print("\n" + "=" * 80)
print("PART 1: VOCABULARY SIZE (FIXED)")
print("=" * 80)

vocab_size = tokenizer.n_vocab
print(f"\nGPT-2 Tokenizer Vocabulary Size: {vocab_size:,}")
print(f"\nThis is FIXED and NEVER changes!")
print(f"Think of it as a dictionary with {vocab_size:,} entries")
print(f"\nSome examples from the vocabulary:")
print(f"  Token ID   0: '{tokenizer.decode([0])}'")
print(f"  Token ID   1: '{tokenizer.decode([1])}'")
print(f"  Token ID  40: '{tokenizer.decode([40])}'")
print(f"  Token ID 367: '{tokenizer.decode([367])}'")
print(f"  Token ID 50256: '{tokenizer.decode([50256])}'")

print("\n" + "=" * 80)
print("PART 2: TOKEN COUNT (VARIES BY TEXT)")
print("=" * 80)

# Example 1: Very short text
text1 = "Hello"
tokens1 = tokenizer.encode(text1)
print(f"\nExample 1: '{text1}'")
print(f"  Characters: {len(text1)}")
print(f"  Token IDs: {tokens1}")
print(f"  Token count: {len(tokens1)}")

# Example 2: Short sentence
text2 = "Hello, how are you today?"
tokens2 = tokenizer.encode(text2)
print(f"\nExample 2: '{text2}'")
print(f"  Characters: {len(text2)}")
print(f"  Token IDs: {tokens2}")
print(f"  Token count: {len(tokens2)}")

# Example 3: Longer text
text3 = "The quick brown fox jumps over the lazy dog. This is a test sentence."
tokens3 = tokenizer.encode(text3)
print(f"\nExample 3: '{text3}'")
print(f"  Characters: {len(text3)}")
print(f"  Token IDs: {tokens3}")
print(f"  Token count: {len(tokens3)}")

# Example 4: Simulate "the-verdict.txt"
text4 = "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera."
tokens4 = tokenizer.encode(text4)
print(f"\nExample 4: Simulated 'the-verdict.txt' excerpt")
print(f"  Characters: {len(text4)}")
print(f"  Token IDs (first 20): {tokens4[:20]}")
print(f"  Token count: {len(tokens4)}")
print(f"\n  If the FULL the-verdict.txt has 20,479 characters,")
print(f"  it would have approximately {len(tokens4) * (20479 / len(text4)):.0f} tokens")
print(f"  (Actual from notebook: 5,145 tokens)")

print("\n" + "=" * 80)
print("PART 3: THE MATH")
print("=" * 80)

print("\nRatio of characters to tokens:")
for i, (text, tokens) in enumerate([(text1, tokens1), (text2, tokens2), (text3, tokens3), (text4, tokens4)], 1):
    ratio = len(text) / len(tokens) if len(tokens) > 0 else 0
    print(f"  Example {i}: {len(text):3d} chars / {len(tokens):2d} tokens = {ratio:.2f} chars/token")

print("\n" + "=" * 80)
print("PART 4: CRITICAL INSIGHT")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ VOCABULARY SIZE (from tokenizer)                               │
│   - Fixed: 50,257 for GPT-2                                    │
│   - Defined when tokenizer was created                         │
│   - NEVER changes regardless of input text                     │
│   - Used for: torch.nn.Embedding(vocab_size, emb_dim)          │
│                                                                 │
│ Example:                                                        │
│   tok_emb = torch.nn.Embedding(50257, 768)                      │
│                                  ^^^^^^                         │
│                            ALWAYS 50,257!                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ TOKEN COUNT (from your text)                                   │
│   - Varies: depends on input text length                       │
│   - Calculated: len(tokenizer.encode(your_text))               │
│   - Changes for every different text                           │
│   - Used for: creating training samples (sliding window)       │
│                                                                 │
│ Examples:                                                       │
│   "Hello"                    →     1 token                     │
│   "Hello, how are you?"      →     6 tokens                    │
│   "the-verdict.txt"          → 5,145 tokens                    │
│   "All of Wikipedia"         → 4 billion tokens                │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("PART 5: WHERE DOES EACH COME FROM?")
print("=" * 80)

print("""
VOCABULARY SIZE:
  Source: tokenizer.n_vocab
  
  Code:
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab  # Returns 50257
    
  Use in model:
    tok_emb = torch.nn.Embedding(vocab_size, emb_dim)
                                 ^^^^^^^^^^
                           This is the lookup table size


TOKEN COUNT:
  Source: len(tokenizer.encode(your_text))
  
  Code:
    raw_text = "your text here..."
    token_ids = tokenizer.encode(raw_text)
    token_count = len(token_ids)  # Varies!
    
  Use in training:
    # Create sliding windows from these tokens
    for i in range(0, token_count - max_length, stride):
        chunk = token_ids[i:i + max_length]
""")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Vocabulary Size (tokenizer): {vocab_size:,} <- FIXED (never changes)
Token Count (text1):         {len(tokens1)} <- VARIES
Token Count (text2):         {len(tokens2)} <- VARIES
Token Count (text3):         {len(tokens3)} <- VARIES
Token Count (text4):         {len(tokens4)} <- VARIES
Token Count (the-verdict):   5,145 <- VARIES

The vocabulary is like a DICTIONARY.
The token count is how many WORDS are in your BOOK.

Different books (texts) have different word counts,
but they all use the SAME dictionary!
""")
