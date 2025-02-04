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
        self.hidden_size = 256
        self.rnn = nn.RNN(input_size=312, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 64)  # Final projection to desired output size

    # Input(312) -> RNN(256 hidden) -> Linear (64 output)
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, features] -> [batch, seq_len=1, features]
        
        # Run RNN
        output, _ = self.rnn(x)  # output shape: [batch, seq_len, hidden_size]
        
        # Get final output
        last_output = output[:, -1, :]  # Take last sequence output
        return self.fc(last_output)  # Project to final dimension

class TowerOne(Tower):
    pass

class TowerTwo(Tower):
    pass

# Initialize the towers
tower_one = TowerOne()
tower_two = TowerTwo()

def calculate_batch_loss(tower_one, tower_two, embeddings_batch, margin):
    query_embeddings, relevant_embeddings, irrelevant_embeddings = embeddings_batch

    # Pass all embeddings through towers
    query_output = tower_one(query_embeddings)
    relevant_output = tower_two(relevant_embeddings)
    irrelevant_output = tower_two(irrelevant_embeddings)

    # Calculate distances
    relevant_distance = 1 - nn.functional.cosine_similarity(query_output, relevant_output, dim=1)
    irrelevant_distance = 1 - nn.functional.cosine_similarity(query_output, irrelevant_output, dim=1)

    # Calculate triplet loss
    triplet_loss = torch.max(torch.tensor(0.0), relevant_distance - irrelevant_distance + margin)
    
    print(f"\nTriplet Loss Analysis:")
    print(f"Relevant distance: {relevant_distance.item():.4f}")
    print(f"Irrelevant distance: {irrelevant_distance.item():.4f}")
    print(f"Triplet loss: {triplet_loss.item():.4f}")

calculate_batch_loss(tower_one, tower_two, embeddings_training_data[0], margin=0.2)
