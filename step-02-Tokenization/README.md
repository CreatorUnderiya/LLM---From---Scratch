# Step-02: Tokenization

## Objective
Convert raw text into tokens and numerical representations for training a Language Model (LLM).

---

## 🧠 What is Tokenization?

Tokenization is the process of breaking down text into smaller units called tokens.

Example:
"I love AI" → ["I", "love", "AI"]

---

## Files

- tokenizer_scratch.py → Tokenization implemented from scratch
- tokenizer_tiktoken.py → Tokenization using tiktoken library
- cleaned.txt → Input text (from Step-01)
- README.md → Documentation

---

## Approach 1: Tokenization From Scratch

### Steps:
- Split text into tokens using regex
- Handle punctuation separately
- Create vocabulary (unique tokens)
- Convert tokens → numerical IDs
- Decode IDs → text

### Key Concepts:
- Word-level tokenization
- Vocabulary creation
- Encoding & decoding

---

## Sample Code (Scratch)

```python
