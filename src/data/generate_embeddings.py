import torch
from .dataset import load_tinybert, get_embeddings
from .load_msmarco import load_msmarco

def generate_and_save_embeddings(train_samples=10, val_samples=5):
    """Generate embeddings for training and validation data and save them."""
    print("Loading TinyBERT model...")
    model, tokenizer = load_tinybert()

    print("\nLoading and processing training data...")
    training_data = load_msmarco(split="train", max_samples=train_samples)
    training_embeddings = get_embeddings(training_data, model, tokenizer)

    print("\nLoading and processing validation data...")
    validation_data = load_msmarco(split="validation", max_samples=val_samples)
    validation_embeddings = get_embeddings(validation_data, model, tokenizer)

    print("\nSaving embeddings...")
    torch.save({
        'training': training_embeddings,
        'validation': validation_embeddings
    }, 'embeddings.pt')
    
    print("Embeddings saved to embeddings.pt")

if __name__ == "__main__":
    generate_and_save_embeddings() 