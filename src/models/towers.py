import torch
import torch.nn as nn
import torch.optim as optim
from src.training.dataloader_train import train_loader, val_loader
import torch.nn.functional as F
import wandb

import sys
import os

# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def cosine_distance(x1, x2):
    return 1 - F.cosine_similarity(x1, x2)

def triplet_loss_function(query, relevant_document, irrelevant_document, distance_function, margin):
    relevant_distance = distance_function(query, relevant_document)
    irrelevant_distance = distance_function(query, irrelevant_document)
    triplet_loss = torch.relu(relevant_distance - irrelevant_distance + margin)
    return triplet_loss.mean()

# Initialize wandb
wandb.init(project="twin-towers-search")

# Log hyperparameters
wandb.config = {
    "learning_rate": 0.01,
    "epochs": 20,
    "batch_size": 32,
    "input_size": 312,
    "hidden_size": 128,
    "output_size": 64,
    "num_layers": 1,
    "margin": 0.3
}

# Define the TwinTowersModel
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure x is 3-dimensional
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a dimension for batch size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def save_parameters(self, rnn_path, fc_path):
        torch.save(self.rnn.state_dict(), rnn_path)
        torch.save(self.fc.state_dict(), fc_path)

    def load_parameters(self, rnn_path, fc_path):
        self.rnn.load_state_dict(torch.load(rnn_path))
        self.fc.load_state_dict(torch.load(fc_path))

class TwinTowersModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TwinTowersModel, self).__init__()
        self.query_tower = RNNModel(input_size, hidden_size, output_size, num_layers)
        self.doc_tower = RNNModel(input_size, hidden_size, output_size, num_layers)

    def forward(self, query, doc):
        query_embedding = self.query_tower(query)
        doc_embedding = self.doc_tower(doc)
        return query_embedding, doc_embedding
    
    def save_parameters(self, query_rnn_path, query_fc_path, doc_rnn_path, doc_fc_path):
        self.query_tower.save_parameters(query_rnn_path, query_fc_path)
        self.doc_tower.save_parameters(doc_rnn_path, doc_fc_path)

    def load_parameters(self, query_rnn_path, query_fc_path, doc_rnn_path, doc_fc_path):
        self.query_tower.load_parameters(query_rnn_path, query_fc_path)
        self.doc_tower.load_parameters(doc_rnn_path, doc_fc_path)


# Could start the file here and import the above from models/towers.py
