"""Contains custom pytorch datasets for training and testing"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tqdm import tqdm

class WikiText103(Dataset):
    """Dataset for benchmark WikiText-103.

    A collection of over 100 million words from wikipedia articles, tokenized

    Args:
        split (str): Dataset split to download, "train", "valid", or "test"
        tokenizer (tokenizers.Tokenizer): Tokenizer to be used
        context_length (int): Number of tokens in each sample
    """
    def __init__(self, split: str, tokenizer: Tokenizer, context_length: int):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.context_length = context_length
        dataset = load_dataset("iohadrubin/wikitext-103-raw-v1")
        tokens = []
        print("Tokenizing dataset...")
        for text in tqdm(dataset[split]["text"]):
            tokens += list(self.tokenizer.encode(text).ids)
        self.data = torch.tensor(tokens, dtype=torch.uint16)
    def __len__(self):
        return (len(self.data)-1)//self.context_length
    def __getitem__(self, idx):
        return self.data[idx*self.context_length:idx*self.context_length+self.context_length].long(), self.data[idx*self.context_length+1:idx*self.context_length+self.context_length+1].long()