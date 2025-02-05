from dataset import get_datasets
import embeddings
from model import DualTowerModel
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

training_dataset = get_datasets("train")

documents = training_dataset.documents

model = DualTowerModel()
model.to(device)
model.eval()
model.load_state_dict(torch.load("weights/two_towers.pth", map_location=device))

tower_two = model.tower_two
tower_two.eval()
tower_two.to(device)

rnn_embeddings = []

batch_size = 256

with torch.no_grad():
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding documents"):
        batch = documents[i:i+batch_size]
        batch_bert_embeddings = embeddings.get_document_embeddings(batch).to(device)
        batch_rnn_embeddings = tower_two.forward(batch_bert_embeddings)
        rnn_embeddings.extend(batch_rnn_embeddings)

embeddings_tensor = torch.stack(rnn_embeddings)
normalized_embeddings = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
torch.save(normalized_embeddings, "weights/doc_embeddings.pth")
