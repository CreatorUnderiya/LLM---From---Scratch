import tiktoken

# Load cleaned text
with open("step-01-data-processing/cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Load tokenizer (GPT-2)
enc = tiktoken.get_encoding("gpt2")

# Encode wala part
tokens = enc.encode(text)

# Decode wala part
decoded_text = enc.decode(tokens)

print("Encoded Tokens:\n", tokens[:50])
print("\nDecoded Text:\n", decoded_text[:200])