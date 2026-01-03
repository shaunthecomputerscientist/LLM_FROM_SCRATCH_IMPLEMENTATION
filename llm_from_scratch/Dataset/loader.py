import tiktoken
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text immediately
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to create training chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True):
    # Initialize the BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create the dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # Create the loader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader