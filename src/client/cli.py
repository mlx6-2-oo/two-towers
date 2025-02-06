import requests
import argparse

def search(query: str, server_url: str, k: int = 5) -> list:
    response = requests.get(
        f"http://{server_url}/search/",
        params={"query": query, "k": k}
    )
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Two Towers Search Client')
    parser.add_argument('--server', default='localhost:8000',
                       help='Server address (default: localhost:8000)')
    args = parser.parse_args()

    print(f"Connecting to server at: {args.server}")
    
    while True:
        query = input("\nQuery (or 'quit'): ")
        if query.lower() == 'quit':
            break
            
        try:
            results = search(query, args.server)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['id']} (score: {r['similarity']:.3f})")
                print(f"{r['document'][:200]}...")
        except requests.RequestException as e:
            print(f"Error: Could not connect to server: {e}") 