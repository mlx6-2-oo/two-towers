import os

import torch
from tqdm import tqdm

import embeddings
from dataset import get_datasets
from model import DualTowerModel
from utils import download_weights, get_device

script_dir = os.path.dirname(os.path.abspath(__file__))

download_weights(
    os.path.join(script_dir, "weights", "doc_embeddings.pth"),
    "https://huggingface.co/datasets/12v12v/ml6-two-towers/resolve/main/doc_embeddings.pth",
)
download_weights(
    os.path.join(script_dir, "weights", "two_towers.pth"),
    "https://huggingface.co/datasets/12v12v/ml6-two-towers/resolve/main/two_towers.pth",
)

device = get_device()

training_dataset = get_datasets("train")

documents = training_dataset.documents

model = DualTowerModel()
model.to(device)
model.eval()
model_path = os.path.join(script_dir, "weights", "two_towers.pth")
model.load_state_dict(torch.load(model_path, map_location=device))

tower_two = model.tower_two
tower_two.eval()
tower_two.to(device)

rnn_embeddings = []

batch_size = 256

with torch.no_grad():
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding documents"):
        batch = documents[i : i + batch_size]
        batch_bert_embeddings = embeddings.get_document_embeddings(batch).to(device)
        batch_rnn_embeddings = tower_two.forward(batch_bert_embeddings)
        rnn_embeddings.extend(batch_rnn_embeddings)

embeddings_tensor = torch.stack(rnn_embeddings)
normalized_embeddings = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
doc_embeddings_path = os.path.join(script_dir, "weights", "doc_embeddings.pth")
torch.save(normalized_embeddings, doc_embeddings_path)
