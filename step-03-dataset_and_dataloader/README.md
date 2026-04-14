# Step-03: Dataset & DataLoader (GPT Style)

##  Objective
Prepare tokenized text into input-target sequences and load them in batches for training a Language Model.

This implementation closely follows GPT-style training pipeline.

---

## Key Concept

After tokenization, raw token IDs cannot be directly fed into a model.

We need:
- Input sequences
- Target sequences (shifted by one position)

---

## Input-Target Example

Text:
I love AI very much

Token IDs:
[10, 25, 90, 12, 45]

Input:
[10, 25, 90, 12]

Target:
[25, 90, 12, 45]

---

##  GPT Style Dataset

We use a sliding window approach to create sequences.

---

##  Code Implementation

```python
from torch.utils.data import Dataset
import torch
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize full text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


## Model Configuration (GPT Style)

We define a configuration similar to GPT architecture:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


Explanation:
vocab_size → total tokens (GPT-2 uses 50257)
context_length → max sequence length
emb_dim → embedding size
n_heads → number of attention heads
n_layers → transformer layers
drop_rate → dropout for regularization
qkv_bias → bias in attention layers


## 🔀 Train-Validation Split

To train and evaluate the model properly, we split the dataset into training and validation parts.

```python
train_ratio = 0.90
split_idx = int(len(text) * train_ratio)

train_data = text[:split_idx]
val_data = text[split_idx:] 