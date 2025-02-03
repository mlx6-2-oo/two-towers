import pandas as pd
import os
from torch.utils.data import Dataset
import embeddings
from tqdm import tqdm
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df



class PairDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        query, document = self.embeddings[idx]
        return query, document
    
class DocumentDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]





training_path = os.path.join(script_dir, "./sources/train.parquet")
test_path = os.path.join(script_dir, "./sources/test.parquet")
validation_path = os.path.join(script_dir, "./sources/validation.parquet")

def get_training_pairs():
    return get_embeddings(training_path)

def get_test_pairs():
    return get_embeddings(test_path)

def get_validation_pairs():
    return get_embeddings(validation_path)

def get_embeddings(path):
    df = load_data(path)

    queries = []
    passages_lists = []
    
    print("Loading data...")
    for _, row in df.iterrows():
        queries.append(row["query"])
        passages_lists.append(row["passages"]["passage_text"].tolist())

    QUERY_BATCH_SIZE = 256
    
    with torch.no_grad():
        # Process queries
        print("Processing queries...")
        query_embeddings = []
        for i in tqdm(range(0, len(queries), QUERY_BATCH_SIZE)):
            batch = queries[i:i + QUERY_BATCH_SIZE]
            batch_embeddings = embeddings.get_query_embeddings(batch)
            query_embeddings.extend(batch_embeddings)

        # Process documents
        print("Processing documents...")
        document_list_embeddings = []
        for passages in tqdm(passages_lists):
            passages_embeddings = embeddings.get_document_embeddings(passages)
            document_list_embeddings.append(passages_embeddings)

    # Create pairs
    pairs = []
    doc_embeddings = []
    for query_embedding, document_embeddings in zip(query_embeddings, document_list_embeddings):
        for doc_embedding in document_embeddings:
            pairs.append((query_embedding.detach().clone(), doc_embedding.detach().clone()))
            doc_embeddings.append(doc_embedding.detach().clone())
    
    # Save embeddings to disk
    print("Saving embeddings to disk...")
    base_path = os.path.splitext(path)[0]
    torch.save(pairs, base_path + "_pairs.pt")
    torch.save(doc_embeddings, base_path + "_docs.pt")
    
    return PairDataset(pairs), DocumentDataset(doc_embeddings)


if __name__ == "__main__":
    validation_df = get_validation_pairs()

    for index, row in validation_df.iterrows():
        if row["query_id"] == 9652:
            print(row["passages"])
