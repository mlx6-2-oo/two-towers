from transformers import AutoTokenizer, BertModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def get_query_embeddings(queries, max_length):
    return get_embeddings(queries, max_length)


def get_document_embeddings(documents, max_length):
    return get_embeddings(documents, max_length)


def get_embeddings(texts, max_length):
    tokens = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    tokens = tokens.to(device)

    embeddings = model(**tokens).last_hidden_state
    return embeddings.cpu()
