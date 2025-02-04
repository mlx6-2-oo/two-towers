import torch
import torch.nn as nn

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

def train_towers(tower_one, tower_two, embeddings_training_data, num_epochs=10, margin=0.2, lr=0.001):
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
