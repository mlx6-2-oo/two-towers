import torch
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoTokenizer, BertModel

from utils import get_device

tokenizer = AutoTokenizer.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D", use_fast=True
)
model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model.eval()

device = get_device()
model.to(device)


def get_query_embeddings(queries):
    return get_embeddings(queries, 24)


def get_document_embeddings(documents):
    return get_embeddings(documents, 256)


def get_embeddings(texts, max_length):
    with torch.no_grad():
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokens.to(device)

        embeddings = model(**tokens).last_hidden_state

        sequence_lengths = torch.sum(tokens["attention_mask"], dim=1).cpu()
        packed_embeddings = pack_padded_sequence(
            embeddings,
            sequence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        return packed_embeddings
