import random
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed



script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
cache_dir = os.path.join(script_dir, 'dataset_cache')

# Load the MS MARCO dataset
dataset = load_dataset("ms_marco", "v1.1", cache_dir=cache_dir)

# Access the training set
data = dataset['train']

n_rows = 1000 # For experimental purposes we only use these rows
train_frac = 0.9

train_size = int(n_rows * train_frac)

# Split the data into training and validation sets
train_data = data.select(range(train_size))
val_data = data.select(range(train_size, n_rows, 1))


def generate_triple(data, index, passage_index):

    query = data['query'][index]
    relevant_document = data['passages'][index]['passage_text'][passage_index]
    # Comment out the line below to generate triples with no irrelevant documents for sampling at later stage
    irrelevant_document = sample_irrelevant(data, index)
    # irrelevant_document = None
    return (query, relevant_document, irrelevant_document)


# all_passage_texts = [passage_text for passage in data["passages"] for passage_text in passage["passage_text"]]
# all_passage_texts = set(all_passage_texts)

# def generate_triples(data):
#     triples = []
#     for idx, items in tqdm(enumerate(data['query'])):
#         for passage_inx in (range(len(data['passages'][idx]['passage_text']))):
#             new_triple = generate_triple(data, idx, passage_inx)
#             triples.append(new_triple)
#     return triples

def generate_triples(data):
    triples = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, items in tqdm(enumerate(data['query'])):
            for passage_inx in range(len(data['passages'][idx]['passage_text'])):
                futures.append(executor.submit(generate_triple, data, idx, passage_inx))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            triples.append(future.result())
    return triples

def generate_random_not_equal_to(exclude, start, end):
        while True:
            rnd = random.randint(start, end)
            if rnd != exclude:
                return rnd

def sample_irrelevant(data, index):
    fake_index = generate_random_not_equal_to(index, 0, len(data["query"])-1)
    fake_number_of_passages = len(data["passages"][fake_index]["passage_text"])
    fake_passage_index = random.randint(0, fake_number_of_passages-1)
    irrelevant_document = data["passages"][fake_index]["passage_text"][fake_passage_index]
 
    return irrelevant_document


# Tokenize and get embeddings:

def load_tinybert():
    """Load TinyBERT model and tokenizer from Hugging Face."""
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    
    return model, tokenizer


def get_embeddings(data, model, tokenizer):
    """Convert text data into embeddings using TinyBERT."""
    embeddings = []
    for query, relevant_passage, irrelevant_passage in data:
        # Get query embeddings
        query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_outputs = model(**query_tokens)
        query_embeddings = query_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token embedding
        
        # Get relevant passage embeddings
        relevant_tokens = tokenizer(relevant_passage, return_tensors="pt", padding=True, truncation=True)
        relevant_outputs = model(**relevant_tokens)
        relevant_embeddings = relevant_outputs.last_hidden_state[:, 0, :]
        
        # Get irrelevant passage embeddings
        irrelevant_tokens = tokenizer(irrelevant_passage, return_tensors="pt", padding=True, truncation=True)
        irrelevant_outputs = model(**irrelevant_tokens)
        irrelevant_embeddings = irrelevant_outputs.last_hidden_state[:, 0, :]
        
        embeddings.append((query_embeddings, relevant_embeddings, irrelevant_embeddings))
    
    return embeddings


# Generate triples
train_data = generate_triples(train_data)
val_data = generate_triples(val_data)

