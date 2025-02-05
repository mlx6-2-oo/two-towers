import torch
import pandas as pd
from ..models.towers import TowerTwo
from .dataset import load_tinybert


def generate_document_embeddings():
    df = pd.read_csv('src/data/id_to_documents.csv')
    model, tokenizer = load_tinybert()  # Get BERT model

    tower_two = TowerTwo()
    tower_two.load_state_dict(torch.load('src/models/tower_two.pt'))
    tower_two.eval()

    embeddings = []
    document_info = []

    with torch.no_grad():
        for id, row in df.iterrows():
            # Get tokens
            tokens = tokenizer(row['passage'], 
                             return_tensors="pt", 
                             padding=True, 
                             truncation=True)
            
            # Get BERT embeddings first (same as in training)
            bert_outputs = model(**tokens)
            bert_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # Just get CLS token embedding
            # Shape: [batch_size, 312]
            
            # Pass BERT CLS embedding to tower_two
            embedding = tower_two(bert_embeddings)
            embeddings.append(embedding)
            document_info.append({'id': row['document_id'], 'document': row['passage']})

    return torch.cat(embeddings), document_info

if __name__ == "__main__":
    embeddings, document_info = generate_document_embeddings()
    torch.save({
        'embeddings': embeddings,
        'document_info': document_info
    }, 'src/models/document_embeddings.pt') 
