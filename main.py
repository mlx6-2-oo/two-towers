import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_datasets
from tqdm import tqdm
import wandb
import embeddings
batch_size = 512
num_epochs = 10
margin = 0.5


torch.manual_seed(7)


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
        self.fc = nn.Linear(256, 64)

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
        self.fc = nn.Linear(512, 64)  # Project down to same size as TowerOne

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


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize model
model = DualTowerModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Get datasets and dataloaders
print("Loading training data...")
triple_dataset = get_datasets("train")
triple_dataloader = DataLoader(triple_dataset, batch_size=batch_size, shuffle=True)

print("Loading validation data...")
val_triple_dataset = get_datasets("validation")
val_triple_dataloader = DataLoader(val_triple_dataset, batch_size=batch_size)

wandb.init(project="two-towers", config={"batch_size": batch_size, "num_epochs": num_epochs, "learning_rate": optimizer.param_groups[0]["lr"]})

# Training loop
for epoch in range(num_epochs):
    model.train()
    batch_losses = []

    train_loop = tqdm(triple_dataloader, 
                     desc=f'Epoch {epoch+1}/{num_epochs}',
                     total=len(triple_dataloader))
    
    for batch_idx, (query_batch, document_batch, neg_document_batch) in enumerate(train_loop):
        model.train()
        optimizer.zero_grad()
        
        # Compute loss
        loss = model.compute_loss(query_batch, document_batch, neg_document_batch)
        
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        train_loop.set_postfix(loss=f'{loss.item():.4f}')
        wandb.log({"batch_loss": loss.item(), "batch": batch_idx + epoch * len(triple_dataloader)})

        if batch_idx % 10 == 0:
            with torch.no_grad():
                # validate with just the first batch
                model.eval()
                val_query_batch, val_documents_batch, val_neg_documents_batch = next(iter(val_triple_dataloader))
                val_loss = model.compute_loss(val_query_batch, val_documents_batch, val_neg_documents_batch)
                wandb.log({"val_loss": val_loss.item()})
    
    train_loss = sum(batch_losses) / len(batch_losses)

    # Validation loop
    val_losses = []
    with torch.no_grad():
        model.eval()
        
        val_loop = tqdm(val_triple_dataloader,
                       desc='Validation',
                       total=len(val_triple_dataloader))
        
        for (val_query_batch, val_documents_batch, val_neg_documents_batch) in val_loop:

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