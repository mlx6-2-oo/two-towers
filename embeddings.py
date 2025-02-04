from transformers import AutoTokenizer, BertModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def get_query_embeddings(queries):
    return get_embeddings(queries)


def get_document_embeddings(documents):
    return get_embeddings(documents)


def get_embeddings(texts):
    tokens = tokenizer(texts, return_tensors="pt", padding=True)
    tokens = tokens.to(device)

    embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.cpu()
