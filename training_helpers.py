import torch
import torch.nn as nn
import wandb

def wandb_init(lr, margin, num_epochs, hidden_size):
    wandb.init(
        project="two-towers",
        name='mode-training',
        config={
            "architecture": "RNN-TinyBERT",
            "learning_rate": lr,
            "margin": margin,
            "num_epochs": num_epochs,
            "hidden_size": hidden_size,
            "output_size": 64
        }
    )

def wandb_log_epoch(epoch, avg_loss, avg_relevant_dist, avg_irrelevant_dist, val_loss):
    metrics = {
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_relevant_distance": avg_relevant_dist,
        "train_irrelevant_distance": avg_irrelevant_dist,
        "train_distance_gap": avg_irrelevant_dist - avg_relevant_dist,
        "val_loss": val_loss
    }

    wandb.log(metrics)

def calculate_batch_loss(tower_one, tower_two, embeddings_batch, margin):
    query_embeddings, relevant_embeddings, irrelevant_embeddings = embeddings_batch

    # Pass all embeddings through towers
    query_output = tower_one(query_embeddings)
    relevant_output = tower_two(relevant_embeddings)
    irrelevant_output = tower_two(irrelevant_embeddings)

    # Calculate distances
    relevant_distance = 1 - nn.functional.cosine_similarity(query_output, relevant_output, dim=1)
    irrelevant_distance = 1 - nn.functional.cosine_similarity(query_output, irrelevant_output, dim=1)

    # Calculate triplet loss
    triplet_loss = torch.max(torch.tensor(0.0), relevant_distance - irrelevant_distance + margin)

    return triplet_loss, relevant_distance.item(), irrelevant_distance.item()

def calculate_validation_loss(tower_one, tower_two, validation_data, margin):
    total_loss = 0
    with torch.no_grad():  # No need to track gradients for validation
        for batch in validation_data:
            loss, _, _ = calculate_batch_loss(tower_one, tower_two, batch, margin)
            total_loss += loss.item()
    return total_loss / len(validation_data)

def train_towers(tower_one, tower_two, embeddings_training_data, validation_data, num_epochs=10, margin=0.2, lr=0.001, use_wandb=False):
    if use_wandb:
        wandb_init(lr, margin, num_epochs, tower_one.hidden_size)

    optimizer = torch.optim.Adam([
        {'params': tower_one.parameters()},
        {'params': tower_two.parameters()}
    ], lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        avg_relevant_dist = 0
        avg_irrelevant_dist = 0

        for batch in embeddings_training_data:
            optimizer.zero_grad()
            loss, rel_dist, irrel_dist = calculate_batch_loss(tower_one, tower_two, batch, margin)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            total_loss += loss.item()
            avg_relevant_dist += rel_dist
            avg_irrelevant_dist += irrel_dist

        # Calculate epoch metrics
        avg_loss = total_loss / len(embeddings_training_data)
        avg_relevant_dist = avg_relevant_dist / len(embeddings_training_data)
        avg_irrelevant_dist = avg_irrelevant_dist / len(embeddings_training_data)
        
        # Calculate validation loss if validation data provided
        val_loss = calculate_validation_loss(tower_one, tower_two, validation_data, margin)
        
        if use_wandb:
            wandb_log_epoch(epoch + 1, avg_loss, avg_relevant_dist, avg_irrelevant_dist, val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
