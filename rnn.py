import torch
import torch.nn as nn
from load_tinybert import load_tinybert

model, tokenizer = load_tinybert()

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

embeddings_training_data = []

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
    
    embeddings_training_data.append((query_embeddings, relevant_embeddings, irrelevant_embeddings))

class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(312, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def forward(self, x):
        return self.layers(x)

class TowerOne(Tower):
    pass

class TowerTwo(Tower):
    pass

# Initialize the towers
tower_one = TowerOne()
tower_two = TowerTwo()

# Process first training example as a test
query_embeddings, relevant_embeddings, _ = embeddings_training_data[0]

# Pass embeddings through towers
query_tower_output = tower_one(query_embeddings)
passage_tower_output = tower_two(relevant_embeddings)

# Calculate similarity score
similarity = nn.functional.cosine_similarity(query_tower_output, passage_tower_output, dim=1)

print(f"\nProcessed first training example:")
print(f"Query: {training_data[0][0]}")
print(f"Relevant passage: {training_data[0][1]}")
print(f"Similarity score: {similarity.item():.4f}")
print(f"Query embedding shape: {query_tower_output.shape}")
print(f"Passage embedding shape: {passage_tower_output.shape}")

