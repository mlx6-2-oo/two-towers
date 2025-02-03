import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_training_pairs, get_validation_pairs
from tqdm import tqdm
import wandb

batch_size = 3
num_epochs = 10


class TowerOne(nn.Module):
    def __init__(self, input_dim=768):  # Default for BERT embeddings
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.model(x)

class TowerTwo(nn.Module):
    def __init__(self, input_dim=768):  # Default for BERT embeddings
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.model(x)

class DualTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower_one = TowerOne()
        self.tower_two = TowerTwo()
    
    def prepare_batch(self, query_batch, document_batch, negative_document_batch):
        query_embeddings = torch.stack([query for query in query_batch]).to(device)
        positive_doc_embeddings = torch.stack([doc for doc in document_batch]).to(device)
        negative_doc_embeddings = torch.stack([doc for doc in negative_document_batch]).to(device)
        return query_embeddings, positive_doc_embeddings, negative_doc_embeddings
        
    def forward(self, queries, positive_docs, negative_docs):
        # Forward pass through towers
        query_output = self.tower_one(queries)
        positive_doc_output = self.tower_two(positive_docs)
        negative_doc_output = self.tower_two(negative_docs)
        
        # Calculate similarities
        positive_score = torch.cosine_similarity(query_output, positive_doc_output)
        negative_score = torch.cosine_similarity(query_output, negative_doc_output)
        
        # Combine scores and create target
        scores = torch.sigmoid(torch.cat([positive_score, negative_score]))
        target = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)])
        
        return scores, target
    
    def compute_loss(self, query_batch, document_batch, negative_document_batch):
        queries, positive_docs, negative_docs = self.prepare_batch(query_batch, document_batch, negative_document_batch)
        scores, target = self(queries, positive_docs, negative_docs)
        return nn.BCELoss()(scores, target)


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize model
model = DualTowerModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters())

# Get datasets and dataloaders
print("Loading training data...")
pairs, documents = get_training_pairs()
pairs_dataloader = DataLoader(pairs, batch_size=batch_size, shuffle=True)
documents_dataloader = DataLoader(documents, batch_size=batch_size, shuffle=True)


print("Loading validation data...")
val_pairs, val_documents = get_validation_pairs()
val_pairs_dataloader = DataLoader(val_pairs, batch_size=batch_size)
val_documents_dataloader = DataLoader(val_documents, batch_size=batch_size)

wandb.init(project="two-towers", config={"batch_size": batch_size, "num_epochs": num_epochs, "learning_rate": optimizer.param_groups[0]["lr"]})

# Training loop
for epoch in range(num_epochs):
    model.train()
    batch_losses = []

    train_loop = tqdm(zip(pairs_dataloader, documents_dataloader), 
                     desc=f'Epoch {epoch+1}/{num_epochs}',
                     total=len(pairs_dataloader))
    
    for batch_idx, ((query_batch, document_batch), neg_document_batch) in enumerate(train_loop):
        optimizer.zero_grad()
        
        # Compute loss
        loss = model.compute_loss(query_batch, document_batch, neg_document_batch)
        
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        train_loop.set_postfix(loss=f'{loss.item():.4f}')
        # wandb.log({"batch_loss": loss.item(), "batch": batch_idx + epoch * len(pairs_dataloader)})
    
    train_loss = sum(batch_losses) / len(batch_losses)

    # Validation loop
    val_losses = []
    with torch.no_grad():
        model.eval()
        
        val_loop = tqdm(zip(val_pairs_dataloader, val_documents_dataloader),
                       desc='Validation',
                       total=len(val_pairs_dataloader))
        
        for (val_query_batch, val_documents_batch), val_neg_documents_batch in val_loop:
            # Compute validation loss
            val_loss = model.compute_loss(val_query_batch, val_documents_batch, val_neg_documents_batch)
            
            val_losses.append(val_loss.item())
            val_loop.set_postfix(loss=f'{val_loss.item():.4f}')

    val_loss = sum(val_losses) / len(val_losses)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

wandb.finish()