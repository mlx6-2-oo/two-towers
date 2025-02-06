import os

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
doc_embeddings_path = os.path.join(script_dir, "weights", "doc_embeddings.pth")
doc_embeddings = torch.load(doc_embeddings_path, map_location=device)

model = DualTowerModel()
model.to(device)
model.eval()
model_path = os.path.join(script_dir, "weights", "two_towers.pth")
model.load_state_dict(torch.load(model_path, map_location=device))

tower_one = model.tower_one
tower_one.eval()
tower_one.to(device)


app = FastAPI()


@app.get("/", response_class=JSONResponse)
def index(query: str):
    print(query)
    query_embedding = embeddings.get_query_embeddings(query).to(device)

    tower_one_output = tower_one.forward(query_embedding)

    normalized_query_embedding = torch.nn.functional.normalize(
        tower_one_output, p=2, dim=1
    )

    cosine_similarities = torch.matmul(normalized_query_embedding, doc_embeddings.T)

    top_k_values, top_k_indices = torch.topk(cosine_similarities, k=5, dim=1)

    response = []

    response = []
    for i in range(len(top_k_values[0])):
        response.append(
            {
                "similarity": top_k_values[0][i].item(),
                "document": documents[top_k_indices[0][i].item()],
            }
        )

    return response
