import os
import argparse
from src.data.generate_embeddings import generate_and_save_embeddings
from src.training.train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run the complete two-tower pipeline')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Step 1: Generate embeddings if they don't exist
    if not os.path.exists('embeddings.pt'):
        print("Generating embeddings...")
        generate_and_save_embeddings(train_samples=10, val_samples=5)
    else:
        print("Using existing embeddings.pt")
    
    # Step 2: Run training
    print("\nStarting training...")
    train_model(use_wandb=args.wandb)

if __name__ == "__main__":
    main() 