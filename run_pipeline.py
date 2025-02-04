import os
from src.data.generate_embeddings import generate_and_save_embeddings

def main():
    # Step 1: Generate embeddings if they don't exist
    if not os.path.exists('embeddings.pt'):
        print("Generating embeddings...")
        generate_and_save_embeddings(train_samples=10, val_samples=5)
    else:
        print("Using existing embeddings.pt")
    
    # Step 2: Run training
    print("\nStarting training...")
    os.system('python train.py')

if __name__ == "__main__":
    main() 