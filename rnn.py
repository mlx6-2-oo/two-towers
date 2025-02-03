from load_tinybert import load_tinybert

model, tokenizer = load_tinybert()

# # Example text to get embeddings for
# text = "The quick brown fox jumps over the lazy dog."

# # Tokenize the text and get model inputs
# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# # Get the model outputs (embeddings)
# outputs = model(**inputs)

# # Get the last hidden states (embeddings)
# embeddings = outputs.last_hidden_state

# # Print information about the embeddings
# print("\nEmbedding information:")
# print(f"Shape of embeddings: {embeddings.shape}")
# print(f"Number of tokens: {embeddings.shape[1]}")
# print(f"Embedding dimension: {embeddings.shape[2]}")

# # Get the embedding for the first token [CLS] which represents the entire sentence
# sentence_embedding = embeddings[0, 0, :]
# print(f"\nFirst token [CLS] embedding (first 5 dimensions):")
# print(sentence_embedding[:5])

# # Get embeddings for each token
# print("\nTokens and their embedding norms:")
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# for i, token in enumerate(tokens):
#     token_embedding = embeddings[0, i, :]
#     embedding_norm = token_embedding.norm().item()
#     print(f"Token: {token:15} Embedding norm: {embedding_norm:.4f}")

training_data = [
    ("What is a good python test library", "Why Pytest is the best python testing library", "Dogs go wild on boozy night out with kitten"),
    ("How to debug memory leaks", "Advanced memory leak detection in Python", "Top 10 pizza recipes for beginners"),
    ("Best practices for API design", "RESTful API design patterns and principles", "Ancient Egyptian burial customs"),
    ("Python async programming guide", "Understanding asyncio and coroutines", "The history of medieval warfare"),
    ("Docker container security", "Securing Docker containers in production", "Best houseplants for your bedroom"),
    ("Machine learning deployment", "MLOps best practices and deployment strategies", "Traditional Irish folk songs"),
    ("Kubernetes vs Docker Swarm", "Container orchestration platforms compared", "Making the perfect sourdough bread"),
    ("GraphQL vs REST APIs", "Modern API architectural patterns", "Top fishing spots in Canada"),
    ("Microservices architecture patterns", "Building resilient microservices", "Ancient Roman cooking techniques"),
    ("Database indexing strategies", "Optimizing database performance", "Training your pet parrot"),
    ("CI/CD pipeline setup", "Automated deployment workflow guide", "History of Renaissance art"),
    ("Web security best practices", "Preventing common web vulnerabilities", "Growing tomatoes in your garden"),
    ("Cloud cost optimization", "Reducing AWS infrastructure costs", "Best hiking trails in Colorado"),
    ("Serverless architecture guide", "AWS Lambda and serverless computing", "Vintage car restoration tips"),
    ("Data streaming with Kafka", "Real-time data processing patterns", "Traditional Japanese tea ceremonies"),
    ("MongoDB vs PostgreSQL", "Choosing the right database for your project", "Basic knitting patterns for beginners")
]

tokenised_training_data = []

for query, relevant_passage, irrelevant_passage in training_data:
    query_input = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    relevant_passage_input = tokenizer(relevant_passage, return_tensors="pt", padding=True, truncation=True)
    irrelevant_passage_input = tokenizer(irrelevant_passage, return_tensors="pt", padding=True, truncation=True)
    tokenised_training_data.append((query_input, relevant_passage_input, irrelevant_passage_input))
