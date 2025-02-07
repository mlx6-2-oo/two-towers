
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_train import train_loader, val_loader
import torch.nn.functional as F
import wandb
from src.models.towers import TwinTowersModel, triplet_loss_function, cosine_distance

import sys
import os

# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Initialize the model, loss function, and optimizer
input_size = 312  # Example input size
hidden_size = 128
output_size = 64
num_layers = 1

model = TwinTowersModel(input_size, hidden_size, output_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
batch_size = 32
margin = 0.3

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        query_tokens, relevant_tokens, irrelevant_tokens = batch
        query_tokens = {key: val.squeeze().to(device).float() for key, val in query_tokens.items()}
        relevant_tokens = {key: val.squeeze().to(device).float() for key, val in relevant_tokens.items()}
        irrelevant_tokens = {key: val.squeeze().to(device).float() for key, val in irrelevant_tokens.items()}
        
        optimizer.zero_grad()
        query_embedding, relevant_embedding = model(query_tokens['input_ids'], relevant_tokens['input_ids'])
        _, irrelevant_embedding = model(query_tokens['input_ids'], irrelevant_tokens['input_ids'])

        # Calculate triplet loss
        loss = triplet_loss_function(query_embedding, relevant_embedding, irrelevant_embedding, cosine_distance, margin)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


# Validation loop
model.eval()
with torch.no_grad():
    running_val_loss = 0.0
    for batch in val_loader:
        query_tokens, relevant_tokens, irrelevant_tokens = batch
        query_tokens = {key: val.squeeze().to(device).float() for key, val in query_tokens.items()}
        relevant_tokens = {key: val.squeeze().to(device).float() for key, val in relevant_tokens.items()}
        irrelevant_tokens = {key: val.squeeze().to(device).float() for key, val in irrelevant_tokens.items()}

        query_embedding, relevant_embedding = model(query_tokens['input_ids'], relevant_tokens['input_ids'])
        _, irrelevant_embedding = model(query_tokens['input_ids'], irrelevant_tokens['input_ids'])

        # Calculate validation triplet loss
        val_loss = triplet_loss_function(query_embedding, relevant_embedding, irrelevant_embedding, cosine_distance, margin)
        running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})
    print(f'Validation Loss: {avg_val_loss:.4f}')

    
import os

# Create the directory if it doesn't exist
save_dir = "/Users/dgwalters/ML Projects/MLX-2/two-towers/saved-parameters"
os.makedirs(save_dir, exist_ok=True)

# Save model parameters
model.save_parameters(
    os.path.join(save_dir, "query_rnn.pth"),
    os.path.join(save_dir, "query_fc.pth"),
    os.path.join(save_dir, "doc_rnn.pth"),
    os.path.join(save_dir, "doc_fc.pth")
)
