import re

class SimpleTokenizer():
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}

    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        vocabs = sorted(list(set(tokens)))

        # add special tokens
        vocabs.append("<UNK>")
        vocabs.append("<END>")

        self.word_to_index = {word: i for i, word in enumerate(vocabs)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}

    def encoder(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        tokens.append("<END>")

        token_ids = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in tokens]
        return token_ids

    def decoder(self, token_ids):
        tokens = [self.index_to_word.get(i, "<UNK>") for i in token_ids]

        if "<END>" in tokens:
            tokens = tokens[:tokens.index("<END>")]

        return " ".join(tokens)


# TEST CODE (IMPORTANT)

with open("step-01-data-processing/cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = SimpleTokenizer()

# Train tokenizer
tokenizer.fit(text)

# Encode
encoded = tokenizer.encoder("This is my first LLM project")

# Decode
decoded = tokenizer.decoder(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)