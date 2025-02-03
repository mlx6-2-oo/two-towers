from transformers import AutoModel, AutoTokenizer

def load_tinybert():
    """
    Load TinyBERT model and tokenizer from Hugging Face.
    """
    # Initialize tokenizer and model from pretrained
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    
    return model, tokenizer

def example_usage(model, tokenizer):
    """
    Demonstrate basic usage of the model.
    """
    # Example text
    text = "Hello, this is a test sentence for TinyBERT."
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get model outputs
    outputs = model(**inputs)
    
    # Print some basic information
    print("\nModel output shape:", outputs.last_hidden_state.shape)
    print("Sequence length:", inputs["input_ids"].shape[1])
    print("Hidden state size:", outputs.last_hidden_state.shape[-1])

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_tinybert()
    
    # Run example
    print("\nRunning example usage...")
    example_usage(model, tokenizer) 