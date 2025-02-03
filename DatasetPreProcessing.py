#import dataset
from datasets import load_dataset
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

rawdata = load_dataset("microsoft/ms_marco", 'v1.1')

#Defining a cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
#
#
# extract first query and each passage
data=[]

for i in range (100):
  uncleaned_query = rawdata['train'][i]['query']
  uncleaned_passages = rawdata['train'][i]['passages']['passage_text']
  query = clean_text(uncleaned_query)
  passages = [clean_text(passage) for passage in uncleaned_passages]
  data.extend({'query': query, 'passage': passage} for passage in passages)
#
#  
#Using BERT for tokenisation and embeddings of queries/passages
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def vectorise_word(word):
    inputs = tokenizer(word, return_tensors='pt', padding=False, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    word_embedding = output.last_hidden_state[0][1]
    return word_embedding
#
#
#Test embedding
print(vectorise_word("test"))


#create dataframe with queries corresponding to each passage
df=pd.DataFrame(data)
# print(df)
#
#
#

