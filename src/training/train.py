import torch
from ..models.towers import TowerOne, TowerTwo
from .trainer import train_towers

def train_model(embeddings_path='embeddings.pt', num_epochs=10, margin=0.2, lr=0.001, use_wandb=False):
    """Train the two-tower model using pre-computed embeddings."""
    # Load pre-computed embeddings
    print("Loading pre-computed embeddings...")
    embeddings = torch.load(embeddings_path)
    embeddings_training_data = embeddings['training']
    embeddings_validation_data = embeddings['validation']

    # Initialize the towers
    tower_one = TowerOne()
    tower_two = TowerTwo()

    # Train the model
    train_towers(
        tower_one=tower_one,
        tower_two=tower_two,
        embeddings_training_data=embeddings_training_data,
        validation_data=embeddings_validation_data,
        num_epochs=num_epochs,
        margin=margin,
        lr=lr,
        use_wandb=use_wandb
    )

if __name__ == "__main__":
    train_model() 