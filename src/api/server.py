from fastapi import FastAPI
from ..inference.search import init_models, search_with_models

app = FastAPI()

# Load models at startup
print("Loading models...")
bert_model, tokenizer, tower_one = init_models()

@app.get("/search/")
async def search_endpoint(query: str, k: int = 5):
    return search_with_models(query, bert_model, tokenizer, tower_one, k) 