from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..inference.search import init_models, search_with_models

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
print("Loading models...")
bert_model, tokenizer, tower_one = init_models()

@app.get("/search/")
async def search_endpoint(query: str, k: int = 5):
    return search_with_models(query, bert_model, tokenizer, tower_one, k) 