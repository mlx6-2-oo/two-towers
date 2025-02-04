import random
from datasets import load_dataset
import os


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
cache_dir = os.path.join(script_dir, 'dataset_cache')

# Load the MS MARCO dataset
dataset = load_dataset("ms_marco", "v2.1", cache_dir=cache_dir)

# Access the training set
data = dataset['train']


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

train_data = generate_triples(data[:100])

# print(test_data_50[-1])
print(f'length of tuples list:{len(train_data)}')
print('*************************')
print(len(train_data[0]))



