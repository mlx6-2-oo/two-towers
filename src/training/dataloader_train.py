import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import sys
import os

# Add the root directory of your project to sys.path (not sure this is relevant)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.generate_triples import train_data, val_data

"""
Build Pytorch Dataset object from database.

Instantiate with triples from generate_triples & tokenizer.


"""
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=312):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, relevant_passage, irrelevant_passage = self.data[idx]
        query_tokens = self.tokenizer(query, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        relevant_tokens = self.tokenizer(relevant_passage, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        irrelevant_tokens = self.tokenizer(irrelevant_passage, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        
        return query_tokens, relevant_tokens, irrelevant_tokens

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Assuming training_data and validation_data are lists of triples (query, relevant_passage, irrelevant_passage)
train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)

# Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

