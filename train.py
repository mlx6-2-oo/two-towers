from src.data.dataset import load_tinybert, get_embeddings, training_data, validation_data
from src.models.towers import TowerOne, TowerTwo
from src.training.trainer import train_towers

def main():
    # Load TinyBERT model and tokenizer
    model, tokenizer = load_tinybert()

    # Get embeddings for training and validation data
    embeddings_training_data = get_embeddings(training_data, model, tokenizer)
    embeddings_validation_data = get_embeddings(validation_data, model, tokenizer)

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