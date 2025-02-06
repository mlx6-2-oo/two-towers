import torch
from torch.utils.data import DataLoader
from dataset import get_datasets
from tqdm import tqdm
import wandb
from model import DualTowerModel, margin

batch_size = 512
num_epochs = 10

torch.manual_seed(7)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize model
model = DualTowerModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Get datasets and dataloaders
print("Loading training data...")
triple_dataset = get_datasets("train")
triple_dataloader = DataLoader(triple_dataset, batch_size=batch_size, shuffle=True)

print("Loading validation data...")
val_triple_dataset = get_datasets("validation")

wandb.init(project="two-towers", config={"batch_size": batch_size, "num_epochs": num_epochs, "learning_rate": optimizer.param_groups[0]["lr"], "margin": margin})

# Training loop
for epoch in range(num_epochs):
    val_triple_dataloader = DataLoader(val_triple_dataset, batch_size=batch_size)
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

        if batch_idx % 100 == 0:
            with torch.no_grad():
                # validate with just the first batch
                model.eval()
                val_query_batch, val_documents_batch, val_neg_documents_batch = next(iter(val_triple_dataloader))
                val_loss = model.compute_loss(val_query_batch, val_documents_batch, val_neg_documents_batch)
                wandb.log({"batch_val_loss": val_loss.item()})
    
    train_loss = sum(batch_losses) / len(batch_losses)

    # save weights and upload to wandb
    torch.save(model.state_dict(), f"two_towers_{epoch}.pth")
    wandb.save(f"two_towers_{epoch}.pth")


    # reset the val_triple_dataloader
    val_triple_dataloader = DataLoader(val_triple_dataset, batch_size=batch_size)
    # Validation loop
    val_losses = []
    with torch.no_grad():
        model.eval()
        
        val_loop = tqdm(val_triple_dataloader,
                       desc='Validation',
                       total=len(val_triple_dataloader))
        

        for batch_idx, (val_query_batch, val_documents_batch, val_neg_documents_batch) in enumerate(val_loop):
            if batch_idx > 10:
                break
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