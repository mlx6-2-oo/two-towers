from datasets import load_dataset
import random

def load_msmarco(split="train", max_samples=None):
    """
    Load the MS MARCO dataset.
    
    Args:
        split (str): Which split to load ("train", "validation", or "test")
        max_samples (int, optional): Maximum number of samples to load. None for all.
    
    Returns:
        list: List of (query, relevant_passage, irrelevant_passage) tuples
    """
    # Load the dataset
    print(f"Loading MS MARCO {split} split...")
    dataset = load_dataset("ms_marco", "v2.1", split=split)
    
    if max_samples:
        dataset = dataset.select(range(max_samples))
    
    # Convert dataset to list for easier processing
    data = {
        'query': list(dataset['query']),
        'passages': list(dataset['passages'])
    }
    
    # Generate triples using the helper functions
    examples = generate_triples(data)
    
    print(f"Loaded {len(examples)} examples from MS MARCO {split} split")
    return examples

def generate_triple(data, index, passage_index):
    query = data['query'][index]
    relevant_document = data['passages'][index]['passage_text'][passage_index]
    # Comment out the line below to generate triples with no irrelevant documents for sampling at later stage
    irrelevant_document = sample_irrelevant(data, index)
    # irrelevant_document = None
    return (query, relevant_document, irrelevant_document)

def generate_triples(data):
    triples = []
    for idx, items in enumerate(data['query']):
        for passage_inx in range(len(data['passages'][idx]['passage_text'])):
            new_triple = generate_triple(data, idx, passage_inx)
            triples.append(new_triple)
    return triples

def sample_irrelevant(data, index):
    all_passage_texts = [passage_text for passage in data["passages"] for passage_text in passage["passage_text"]]
    current_passage_texts = data["passages"][index]["passage_text"]
    sample_space = list(set(all_passage_texts) - set(current_passage_texts))
    rnd_response = random.choice(sample_space)
    return rnd_response

if __name__ == "__main__":
    # Test the loader with a small sample
    print('about to load data')
    examples = load_msmarco(split="train", max_samples=5)
    
    # Print a sample
    print("\nExample triple:")
    query, rel, irrel = examples[0]
    print(f"Query: {query}")
    print(f"Relevant: {rel[:100]}...")
    print(f"Irrelevant: {irrel[:100]}...") 