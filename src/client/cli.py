import requests

def search(query: str, k: int = 5) -> list:
    response = requests.get(
        "http://localhost:8000/search/",
        params={"query": query, "k": k}
    )
    return response.json()

if __name__ == "__main__":
    while True:
        query = input("\nQuery (or 'quit'): ")
        if query.lower() == 'quit':
            break
            
        try:
            results = search(query)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['id']} (score: {r['similarity']:.3f})")
                print(f"{r['document'][:200]}...")
        except requests.RequestException as e:
            print(f"Error: Could not connect to server: {e}") 