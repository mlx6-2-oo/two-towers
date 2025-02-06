import os
import random

from datasets import load_dataset
from torch.utils.data import Dataset

import embeddings
from utils import get_device

script_dir = os.path.dirname(os.path.abspath(__file__))

device = get_device()

dataset = load_dataset("microsoft/ms_marco", "v1.1")


def get_datasets(dataset_name):
    df = dataset[dataset_name].to_pandas()
    queries = []
    documents = []
    for query, passages in zip(df["query"].tolist(), df["passages"].tolist()):
        for passage in passages["passage_text"]:
            queries.append(query)
            documents.append(passage)

    return TripleDataset(queries, documents)


class TripleDataset(Dataset):
    def __init__(self, queries, documents):
        self.queries = queries
        self.documents = documents

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # return query, document, random document which is at least 10 away from idx and isn't the same as document
        random_idx = random.randint(0, len(self.documents) - 1)
        while (
            random_idx == idx
            or abs(random_idx - idx) < 10
            or self.documents[random_idx] == self.documents[idx]
        ):
            random_idx = random.randint(0, len(self.documents) - 1)
        return (
            self.queries[idx],
            self.documents[idx],
            self.documents[random_idx],
        )

    def collate_fn(self, batch):
        queries, documents, random_documents = zip(*batch)

        query_embedding = embeddings.get_query_embeddings(queries).to(device)
        document_embedding = embeddings.get_document_embeddings(documents).to(device)
        random_document_embedding = embeddings.get_document_embeddings(
            random_documents
        ).to(device)

        return query_embedding, document_embedding, random_document_embedding


if __name__ == "__main__":
    pass
