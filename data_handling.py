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

def get_embeddings(training_data, model, tokenizer):
    embeddings = []
    for query, relevant_passage, irrelevant_passage in training_data:
        # Get query embeddings
        query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_outputs = model(**query_tokens)
        query_embeddings = query_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token embedding
        
        # Get relevant passage embeddings
        relevant_tokens = tokenizer(relevant_passage, return_tensors="pt", padding=True, truncation=True)
        relevant_outputs = model(**relevant_tokens)
        relevant_embeddings = relevant_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token embedding
        
        # Get irrelevant passage embeddings
        irrelevant_tokens = tokenizer(irrelevant_passage, return_tensors="pt", padding=True, truncation=True)
        irrelevant_outputs = model(**irrelevant_tokens)
        irrelevant_embeddings = irrelevant_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token embedding
        
        embeddings.append((query_embeddings, relevant_embeddings, irrelevant_embeddings))

    return embeddings