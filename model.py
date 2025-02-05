import torch
import torch.nn as nn
import embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

margin = 0.3

class TowerOne(nn.Module):
    def __init__(self, input_dim=312):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.0  # Less dropout for shorter sequences
        )
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        # x shape: (batch_size, ~20, bert_dim) for queries
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

class TowerTwo(nn.Module):
    def __init__(self, input_dim=312):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=512,  # Larger hidden size for longer sequences
            num_layers=3,     # More layers to capture document structure
            batch_first=True,
            dropout=0.0       # More dropout for longer sequences
        )
        self.fc = nn.Linear(512, 128)  # Project down to same size as TowerOne

    def forward(self, x):
        # x shape: (batch_size, ~200, bert_dim) for documents
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

class DualTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower_one = TowerOne()
        self.tower_two = TowerTwo()
    
    def prepare_batch(self, query_batch, document_batch, negative_document_batch):
        query_embeddings = embeddings.get_query_embeddings(query_batch).to(device)
        positive_doc_embeddings = embeddings.get_document_embeddings(document_batch).to(device)
        negative_doc_embeddings = embeddings.get_document_embeddings(negative_document_batch).to(device)

        return query_embeddings, positive_doc_embeddings, negative_doc_embeddings
        
    def forward(self, queries, positive_docs, negative_docs):
        # Forward pass through towers
        query_output = self.tower_one(queries)
        positive_doc_output = self.tower_two(positive_docs)
        negative_doc_output = self.tower_two(negative_docs)
        
        # Calculate similarities
        relevant_similarity = torch.cosine_similarity(query_output, positive_doc_output)
        irrelevant_similarity = torch.cosine_similarity(query_output, negative_doc_output)

        # Translate to [0, 2]
        relevant_distance = 1 - relevant_similarity 
        irrelevant_distance = 1 - irrelevant_similarity

        loss = torch.max(torch.tensor(0.0), relevant_distance - irrelevant_distance + margin).mean()
        return loss
        
    def compute_loss(self, query_batch, document_batch, negative_document_batch):
        queries, positive_docs, negative_docs = self.prepare_batch(query_batch, document_batch, negative_document_batch)
        return self(queries, positive_docs, negative_docs)