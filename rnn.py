from load_tinybert import load_tinybert

model, tokenizer = load_tinybert()

# Example text to get embeddings for
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text and get model inputs
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get the model outputs (embeddings)
outputs = model(**inputs)

# Get the last hidden states (embeddings)
embeddings = outputs.last_hidden_state

# Print information about the embeddings
print("\nEmbedding information:")
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Number of tokens: {embeddings.shape[1]}")
print(f"Embedding dimension: {embeddings.shape[2]}")

# Get the embedding for the first token [CLS] which represents the entire sentence
sentence_embedding = embeddings[0, 0, :]
print(f"\nFirst token [CLS] embedding (first 5 dimensions):")
print(sentence_embedding[:5])

# Get embeddings for each token
print("\nTokens and their embedding norms:")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for i, token in enumerate(tokens):
    token_embedding = embeddings[0, i, :]
    embedding_norm = token_embedding.norm().item()
    print(f"Token: {token:15} Embedding norm: {embedding_norm:.4f}")

