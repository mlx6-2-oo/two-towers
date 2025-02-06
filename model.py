import torch
import torch.nn as nn

from utils import get_device

device = get_device()

margin = 0.3


class TowerOne(nn.Module):
    def __init__(self, input_dim=312):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,  # Less dropout for shorter sequences
        )
        self.layer_norm = nn.LayerNorm(256)
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        # x shape: (batch_size, ~20, bert_dim) for queries
        batch_size, seq_len, _ = x.shape
        # Flatten the sequence and feature dimension
        x = x.view(batch_size * seq_len, -1)
        x = self.batch_norm(x)
        x = x.view(batch_size, seq_len, -1)
        _, hidden = self.gru(x)
        hidden = self.layer_norm(hidden[-1])
        return self.fc(hidden)


class TowerTwo(nn.Module):
    def __init__(self, input_dim=312):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=512,  # Larger hidden size for longer sequences
            num_layers=3,  # More layers to capture document structure
            batch_first=True,
            dropout=0.3,  # More dropout for longer sequences
        )
        self.layer_norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 128)  # Project down to same size as TowerOne

    def forward(self, x):
        # x shape: (batch_size, ~200, bert_dim) for documents
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        x = self.batch_norm(x)
        x = x.view(batch_size, seq_len, -1)
        _, hidden = self.gru(x)
        hidden = self.layer_norm(hidden[-1])
        return self.fc(hidden)


class DualTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower_one = TowerOne()
        self.tower_two = TowerTwo()

    def forward(self, queries, positive_docs, negative_docs):
        # Forward pass through towers
        query_output = self.tower_one(queries)
        positive_doc_output = self.tower_two(positive_docs)
        negative_doc_output = self.tower_two(negative_docs)

        return query_output, positive_doc_output, negative_doc_output

    def compute_loss(self, query_batch, document_batch, neg_document_batch):
        query_output, positive_doc_output, negative_doc_output = self(
            query_batch, document_batch, neg_document_batch
        )

        # Calculate similarities
        relevant_similarity = torch.cosine_similarity(query_output, positive_doc_output)
        irrelevant_similarity = torch.cosine_similarity(
            query_output, negative_doc_output
        )

        # Translate to [0, 2]
        relevant_distance = 1 - relevant_similarity
        irrelevant_distance = 1 - irrelevant_similarity

        unclamped_loss = relevant_distance - irrelevant_distance + margin

        loss = torch.max(torch.tensor(0.0), unclamped_loss).mean()
        return (
            loss,
            unclamped_loss.mean(),
            relevant_distance.mean(),
            irrelevant_distance.mean(),
        )
