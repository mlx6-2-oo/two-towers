from transformers import AutoModel, AutoTokenizer

def load_tinybert():
    """Load TinyBERT model and tokenizer from Hugging Face."""
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    
    return model, tokenizer

def get_embeddings(data, model, tokenizer):
    """Convert text data into embeddings using TinyBERT."""
    embeddings = []
    for query, relevant_passage, irrelevant_passage in data:
        # Get query embeddings
        query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_outputs = model(**query_tokens)
        query_embeddings = query_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token embedding
        
        # Get relevant passage embeddings
        relevant_tokens = tokenizer(relevant_passage, return_tensors="pt", padding=True, truncation=True)
        relevant_outputs = model(**relevant_tokens)
        relevant_embeddings = relevant_outputs.last_hidden_state[:, 0, :]
        
        # Get irrelevant passage embeddings
        irrelevant_tokens = tokenizer(irrelevant_passage, return_tensors="pt", padding=True, truncation=True)
        irrelevant_outputs = model(**irrelevant_tokens)
        irrelevant_embeddings = irrelevant_outputs.last_hidden_state[:, 0, :]
        
        embeddings.append((query_embeddings, relevant_embeddings, irrelevant_embeddings))
    
    return embeddings

# Example training and validation data
training_data = [
    ("What is a good python test library", "Why Pytest is the best python testing library", "Dogs go wild on boozy night out with kitten"),
    ("How to debug memory leaks", "Advanced memory leak detection in Python", "Top 10 pizza recipes for beginners"),
    ("Best practices for API design", "RESTful API design patterns and principles", "Ancient Egyptian burial customs"),
    # ... rest of training data
]

validation_data = [
    ("How to optimize Python code", "Advanced Python performance tuning guide", "Making homemade ice cream"),
    ("Git branching strategies", "Git workflow best practices for teams", "Popular bird watching locations"),
    ("SQL query optimization", "Database query performance tuning", "Beginner's guide to origami"),
    # ... rest of validation data
]

# # Load real training and validation data from MS MARCO
# training_data = load_msmarco(split="train", max_samples=1000)  # Adjust max_samples as needed
# validation_data = load_msmarco(split="validation", max_samples=100)