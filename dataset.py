import os
from torch.utils.data import Dataset
import embeddings
from tqdm import tqdm
import torch
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))


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

def download_dataset():
    return load_dataset("microsoft/ms_marco", "v1.1")

def get_training_pairs():
    dataset = download_dataset()
    training_df = dataset["train"].to_pandas()
    return get_embeddings(training_df, "training")

def get_test_pairs():
    dataset = download_dataset()
    test_df = dataset["test"].to_pandas()
    return get_embeddings(test_df, "test")

def get_validation_pairs():
    dataset = download_dataset()
    validation_df = dataset["validation"].to_pandas()
    return get_embeddings(validation_df, "validation")

def get_embeddings(df, name):
    queries = []
    passages_lists = []
    
    print("Loading data...")
    for _, row in df.iterrows():
        queries.append(row["query"])
        passages_lists.append(row["passages"]["passage_text"].tolist())

    QUERY_BATCH_SIZE = 256

    # Find max length of queries
    max_query_length = max(len(query.split()) for query in queries)
    print(f"Max query length: {max_query_length}")

    # Find max length of passages
    max_passage_length = max(
        max(len(passage.split()) for passage in passages)
        for passages in passages_lists
    )
    print(f"Max passage length: {max_passage_length}")
    
    with torch.no_grad():
        # Process queries
        print("Processing queries...")
        query_embeddings = []
        for i in tqdm(range(0, len(queries), QUERY_BATCH_SIZE)):
            batch = queries[i:i + QUERY_BATCH_SIZE]
            batch_embeddings = embeddings.get_query_embeddings(batch, max_query_length)
            query_embeddings.extend(batch_embeddings)

        # Process documents
        print("Processing documents...")
        document_list_embeddings = []
        for passages in tqdm(passages_lists):
            passages_embeddings = embeddings.get_document_embeddings(passages, max_passage_length)
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
    base_path = os.path.join(script_dir, f"./sources/{name}_embeddings")
    torch.save(pairs, base_path + "_pairs.pt")
    torch.save(doc_embeddings, base_path + "_docs.pt")
    
    return PairDataset(pairs), DocumentDataset(doc_embeddings)


if __name__ == "__main__":
    validation_df = get_validation_pairs()

    for index, row in validation_df.iterrows():
        if row["query_id"] == 9652:
            print(row["passages"])
