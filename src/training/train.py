import torch
import argparse
from ..models.towers import TowerOne, TowerTwo
from .trainer import train_towers

def parse_args():
    parser = argparse.ArgumentParser(description='Train the two-tower model')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    return parser.parse_args()

def train_model(use_wandb=False):
    """Train the two-tower model using pre-computed embeddings."""
    # Load pre-computed embeddings
    print("Loading pre-computed embeddings...")
    embeddings = torch.load('embeddings.pt')
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
        num_epochs=10,
        margin=0.2,
        lr=0.001,
        use_wandb=use_wandb
    )

if __name__ == "__main__":
    args = parse_args()
    train_model(use_wandb=args.wandb) 