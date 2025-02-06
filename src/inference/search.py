import torch
import torch.nn.functional as F
from ..models.towers import TowerOne
from ..data.dataset import load_tinybert

def init_models():
    """Initialize models once at startup."""
    # Load BERT and tokenizer
    bert_model, tokenizer = load_tinybert()
    
    # Load tower one
    tower_one = TowerOne()
    tower_one.load_state_dict(torch.load('src/models/tower_one.pt'))
    tower_one.eval()
    
    return bert_model, tokenizer, tower_one

def search_with_models(query, bert_model, tokenizer, tower_one, k=5):
    """Search using pre-loaded models."""
    # Load pre-computed document embeddings
    data = torch.load('src/models/document_embeddings.pt')
    
    # Get query embedding
    with torch.no_grad():
        tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        bert_outputs = bert_model(**tokens)
        bert_embedding = bert_outputs.last_hidden_state[:, 0, :]
        query_embedding = tower_one(bert_embedding)
    
    # Find k nearest neighbors
    similarities = F.cosine_similarity(query_embedding, data['embeddings'], dim=1)
    top_k = torch.topk(similarities, k=k)
    
    return [
        {'similarity': s.item(), **data['document_info'][i.item()]}
        for s, i in zip(top_k.values, top_k.indices)
    ]

def search(query, k=5):
    """Legacy search function that loads models each time."""
    bert_model, tokenizer, tower_one = init_models()
    return search_with_models(query, bert_model, tokenizer, tower_one, k)

if __name__ == "__main__":
    # For CLI usage, load models once
    print("Loading models...")
    bert_model, tokenizer, tower_one = init_models()
    
    while True:
        query = input("\nQuery (or 'quit'): ")
        if query.lower() == 'quit':
            break
            
        results = search_with_models(query, bert_model, tokenizer, tower_one)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['id']} (score: {r['similarity']:.3f})")
            print(f"{r['document'][:200]}...") 