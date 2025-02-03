#import dataset
from datasets import load_dataset
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

#Directly load the dataset as rawdata
rawdata = load_dataset("microsoft/ms_marco", 'v1.1')

#Defining a cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


#Using BERT for tokenisation and embeddings of queries/passages
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


#Defining a function to take a word and then turn it into the pre-defined embeddings from BERT
def vectorise_word(word):
    inputs = tokenizer(word, return_tensors='pt', padding=False, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    word_embedding = output.last_hidden_state.squeeze(0)
    #checking for subwords, mostly written by ChatGPT, annoying
    tokens = tokenizer.tokenize(word)
    word_vectors = []
    current_word_embedding = None
    for token, embedding in zip(tokens, word_embedding):
        # Check if the token is a subword token (starts with '##')
        if token.startswith('##'):
            # Add to the current word's embedding
            current_word_embedding += embedding
        else:
            # If there's a previous word's embedding, save it
            if current_word_embedding is not None:
                word_vectors.append(current_word_embedding)
            # Start a new word embedding
            current_word_embedding = embedding
    if current_word_embedding is not None:
        word_vectors.append(current_word_embedding)
    return [embedding.numpy().tolist() for embedding in word_vectors]
    
# Iterates through the rawdata dataframe up to defined range   
data=[]
for i in range (5):
  #Takes query and passages from row 
  uncleaned_query = rawdata['train'][i]['query']
  uncleaned_passages = rawdata['train'][i]['passages']['passage_text']
  #Cleans and then vectorises each query and passage in row
  query = vectorise_word(clean_text(uncleaned_query))
  passages = [vectorise_word(clean_text(passage)) for passage in uncleaned_passages]
  #Appends the vectorised data to the object data and then iterates to next row
  data.extend({'query': query, 'passage': passage} for passage in passages)

#Stores data in a dataframe called df
df=pd.DataFrame(data)  
first_row = df.iloc[3]
print(first_row)