import torch
from src.models.towers import TowerOne, TowerTwo
from src.training.trainer import train_towers

def main():
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
        use_wandb=False
    )

if __name__ == "__main__":
    main() 