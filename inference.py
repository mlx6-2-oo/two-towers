from dataset import get_datasets
import embeddings
from model import DualTowerModel
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

training_dataset = get_datasets("train")

documents = training_dataset.documents
doc_embeddings = torch.load("weights/doc_embeddings.pth", map_location=device)

model = DualTowerModel()
model.to(device)
model.eval()
model.load_state_dict(torch.load("weights/two_towers.pth", map_location=device))

tower_one = model.tower_one
tower_one.eval()
tower_one.to(device)


app = FastAPI()

@app.get("/", response_class=JSONResponse)
def index(query: str):
    print(query)
    query_embedding = embeddings.get_query_embeddings(query).to(device)

    tower_one_output = tower_one.forward(query_embedding)

    normalized_query_embedding = torch.nn.functional.normalize(tower_one_output, p=2, dim=1)

    cosine_similarities = torch.matmul(normalized_query_embedding, doc_embeddings.T)

    top_k_values, top_k_indices = torch.topk(cosine_similarities, k=5, dim=1)

    response = []

    response = []
    for i in range(len(top_k_values[0])):
        response.append({
            "similarity": top_k_values[0][i].item(),
            "document": documents[top_k_indices[0][i].item()]
        })

    return response
