from fastapi import FastAPI
from ..inference.search import search

app = FastAPI()

@app.get("/search/")
async def search_endpoint(query: str, k: int = 5):
    return search(query, k) 