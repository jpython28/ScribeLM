import requests
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class WikiText2(Dataset):
  def __init__(self, split: str, tokenizer: Tokenizer, context_length: int):
    super().__init__()
    self.split = split
    self.tokenizer = tokenizer
    self.context_length = context_length

    if self.split == "train":
      url = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.train.raw"
    elif self.split == "test":
      url = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.test.raw"
    elif self.split == "valid":
      url = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.valid.raw"
    else:
      raise ValueError("split must be either \"train\" or \"valid\" or \"test\"")
    
    response = requests.get(url)
    text = response.text

    assert response.status_code==200, f"Error retrieving text, status code {response.status_code}"

    self.data = torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)

  def __len__(self):
    return len(self.data)-self.context_length
  
  def __getitem__(self, idx):
    return self.data[idx:idx+self.context_length], self.data[idx+1:idx+self.context_length+1]