
import torch 
import tiktoken
from torch.utils.data import DataLoader ,Dataset 

with open("cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

class GPTDataset(Dataset):
  def __init__(self,text,tokenizer,max_length, stride):
    self.input_ids = []
    self.target_ids = []

    # tokenize the entire text
    token_ids = tokenizer.encode(text,allowed_special={"<|endoftext|>"})

    # Use a sliding window to chunk the book into overlapping sequences of max_length

    for i in range(0,len(token_ids)-max_length , stride):
      input_chunk = token_ids[i:i+max_length]
      target_chunk = token_ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self,idx):
    return self.input_ids[idx], self.target_ids[idx]

# create Dataloader
def create_dataloader(text, batch_size=4, max_length=256,stride=128, shuffle=True,drop_last=True, num_workers=0):

  #initialize the tokenizer

  tokenizer = tiktoken.get_encoding("gpt2")

  # create dataset
  dataset = GPTDataset(text,tokenizer, max_length, stride)

  # create dataloader

  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      num_workers=num_workers  # it is used for parallel processing
  )

  return dataloader


# We define a configuration similar to GPT architecture:

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}


# split
train_ratio = 0.90
split_idx = int(len(text) * train_ratio)

train_data = text[:split_idx]
val_data = text[split_idx:]


# create Dataloader
def create_dataloader(text, batch_size=4, max_length=256,stride=128, shuffle=True,drop_last=True, num_workers=0):

  #initialize the tokenizer

  tokenizer = tiktoken.get_encoding("gpt2")

  # create dataset
  dataset = GPTDataset(text,tokenizer, max_length, stride)

  # create dataloader

  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      num_workers=num_workers  # it is used for parallel processing
  )

  return dataloader



# loaders
train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=256,
    stride=128,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=256,
    stride=128,
    drop_last=False,
    shuffle=False,
    num_workers=0
)