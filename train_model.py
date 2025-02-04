import torch
import torch.nn as nn
from data_handling import load_tinybert, get_embeddings
from rnn import TowerOne, TowerTwo

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

embeddings_training_data = get_embeddings(training_data, model, tokenizer)

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

    return triplet_loss

def train_towers(tower_one, tower_two, embeddings_training_data, num_epochs=10, margin=0.2):
    optimizer = torch.optim.Adam([
        {'params': tower_one.parameters()},
        {'params': tower_two.parameters()}
    ], lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in embeddings_training_data:
            optimizer.zero_grad()
            loss = calculate_batch_loss(tower_one, tower_two, batch, margin)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(embeddings_training_data):.4f}")

# Initialize the towers
tower_one = TowerOne()
tower_two = TowerTwo()

train_towers(tower_one, tower_two, embeddings_training_data, 10, 0.2)
