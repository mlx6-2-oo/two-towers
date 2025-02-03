from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_query_embeddings(queries):
    return get_embeddings(queries)


def get_document_embeddings(documents):
    return get_embeddings(documents)


def get_embeddings(texts):
    tokens = tokenizer(texts, return_tensors="pt", padding=True)

    embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings
