import torch
import torch.nn.functional as F
from ..models.towers import TowerOne
from ..data.dataset import load_tinybert

def search(query, k=5):
    # Load models and data
    model, tokenizer = load_tinybert()
    tower_one = TowerOne()
    tower_one.load_state_dict(torch.load('src/models/tower_one.pt'))
    tower_one.eval()
    
    data = torch.load('src/models/document_embeddings.pt')
    
    # Get query embedding (same pipeline as in training)
    with torch.no_grad():
        tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        bert_outputs = model(**tokens)
        bert_embedding = bert_outputs.last_hidden_state[:, 0, :]
        query_embedding = tower_one(bert_embedding)
    
    # Find k nearest neighbors using cosine similarity
    similarities = F.cosine_similarity(query_embedding, data['embeddings'], dim=1)
    top_k = torch.topk(similarities, k=k)
    
    return [
        {'similarity': s.item(), **data['document_info'][i.item()]}
        for s, i in zip(top_k.values, top_k.indices)
    ]

if __name__ == "__main__":
    while True:
        query = input("\nQuery (or 'quit'): ")
        if query.lower() == 'quit':
            break
            
        results = search(query)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['id']} (score: {r['similarity']:.3f})")
            print(f"{r['document'][:200]}...") 