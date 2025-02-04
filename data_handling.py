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
