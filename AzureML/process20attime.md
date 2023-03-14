# Azure ML - Python process 20 rows at a time with Azure Open AI

## Process large dataframe by chunks of 20

## Pre-requisites

- Azure Account
- Storage account
- Azure machine learning
- Azure open ai service

## Goal

- Azure Open AI is a service that allows you to use GPT-3 to generate text.
- But there is limitation on how much we can send at a time
- At the time of this document it was 20 request/sec
- So we will process 20 rows at a time from pandas dataframe

## Code

- import libraries

```
from pdfreader import SimplePDFViewer
from typing import Container
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from azure.storage.blob import ResourceTypes, AccountSasPermissions
from azure.storage.blob import generate_account_sas    
from datetime import *
```

- Read the data

```
df = pd.read_csv('alldatatext.csv')
```

- find total row count

```
total = df.count()
```

- strip uncessary space and NA

```
df1 = df['text'].str.strip().dropna()
```

```
df1 = df[(df.text != '')]
```

- final total

```
total = df1.count()
```

- bring opean ai 

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoiservicenow.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxx"
```

- import for tokenization

```
import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
```

- calculate tokens

```
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df1['n_tokens'] = df1["text"].apply(lambda x: len(tokenizer.encode(x)))
df1 = df1[df1.n_tokens<2000]
len(df1)
```

- Create the summary function

```
def getsummary(mystring):
    response = openai.Completion.create(
    engine="davinci003",
    prompt= 'Summarize ' + mystring,
    temperature=0.9,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1
    )
    return response.choices[0].text
```

- configure 

```
chunksize = 20
start = 0
end = total
print(end)
```

- display the column

```
dffinal.columns
```

- process dataframe in chunks

```
for i in range(start, len(df1), chunksize) :
    #display(df1.iloc[i:chunksize])
    df2 = df1.iloc[int(i):int(chunksize + i)].copy()
    
    df2['summary'] = df2["text"].apply(lambda x : getsummary(x))
    #display(df2)
    df2.to_csv('datawithsummary1.csv', mode='a', index=False, header=False)
    print(i)
    
    i = i + chunksize
    #print(i)
```

- Read the file saved

```
df3 = pd.read_csv('datawithsummary1.csv', header=None)
```

- Assign column name

```
df3.columns = ['text','n_tokens','summary']
```

- Display and see if all data with summarization is available

```
display(df3)
```