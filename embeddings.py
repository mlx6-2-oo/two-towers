from transformers import AutoTokenizer, BertModel
import torch

tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', use_fast=True)
model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def get_query_embeddings(queries):
    return get_embeddings(queries, 24)


def get_document_embeddings(documents):
    return get_embeddings(documents, 256)


def get_embeddings(texts, max_length):
    with torch.no_grad():
        tokens = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        tokens = tokens.to(device)

        embeddings = model(**tokens).last_hidden_state
        return embeddings.cpu()
