# Azure Open AI Summarize in pandas dataframe

## Overview

- Summarize the data in pandas dataframe
- Load PDF data into pandas dataframe
- Clean the data
- Load all pages into one row per pdf
- Using Azure Machine learning
- Load the pdf in a blob container

## Code

- Pip install pdfreader
- Load pip reader

```
pip install pdfreader
```

```
from pdfreader import SimplePDFViewer
```

- Import storage libraries

```
from typing import Container
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from azure.storage.blob import ResourceTypes, AccountSasPermissions
from azure.storage.blob import generate_account_sas    
from datetime import *

today = str(datetime.now().date())
print(today)
```

- Setup connection string
- Setup the key

```
# Source Client
connection_string = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' # The connection string for the source container
account_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx' # The account key for the source container
# source_container_name = 'newblob' # Name of container which has blob to be copied

# Create client
client = BlobServiceClient.from_connection_string(connection_string) 
```

- initialize the container

```
client = BlobServiceClient.from_connection_string(connection_string)
all_containers = client.list_containers(include_metadata=True)
```

- Create a empty dataframe

```
import pandas as pd

df = pd.DataFrame()
```

- Loop the files in container
- Load the data into pdf
- load into dataframe

```
container_client = client.get_container_client("containername")
from azure.storage.blob import BlobClient

# print(container_client)
blobs_list = container_client.list_blobs()
for blob in blobs_list:
    # Create blob client for source blob
    source_blob = BlobClient(
    client.url,
    container_name = "acccsuite"
    , blob_name = blob.name
    #,credential = sas_token
    )
    #print(blob.name)
    filename = blob.name
    blob = BlobClient.from_connection_string(conn_str=connection_string, container_name="containername", blob_name=blob.name)

    with open(filename, "wb") as my_blob:
        blob_data = blob.download_blob()
        #blob_data.readinto(my_blob)
        data = blob_data.readall()
        #print(data)
        #fd = open(blob.name, "rb")
        viewer = SimplePDFViewer(data)
        all_pages = [p for p in viewer.doc.pages()]
        number_of_pages = len(all_pages)
        page_strings = ""
        #print(number_of_pages)
        for page_number in range(1, number_of_pages + 1):
            viewer.navigate(int(page_number))
            viewer.render()
            page_strings += " ".join(viewer.canvas.strings).replace('     ', '\n\n').strip()
            #print(f'Current Page Number: {page_number}')
            #print(f'Page Text: {page_strings}')

        if len(page_strings) > 0:
            df = df.append({ 'text' : page_strings}, ignore_index = True)    
```

- Check the count of dataframe

```
df.count()
```

- Setup open ai services

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://servicename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxxxxxxxxxxxxxx"
```

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

- Now create a columns with token

```
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df1['n_tokens'] = df1["text"].apply(lambda x: len(tokenizer.encode(x)))
df1 = df1[df1.n_tokens<2000]
len(df1)
```

- now split the data into 2 chunks of each 20

```
dfcontent = df1.iloc[:20].copy()
dfcontent1 = df1.iloc[20:40].copy()
dfcontent2 = df1.iloc[40:].copy()
```

- Now create a function to summarize the data

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

- Now apply the function to the dataframe

```
dfcontent['summary'] = dfcontent["text"].apply(lambda x : getsummary(x))
```

```
dfcontent1['summary'] = dfcontent1["text"].apply(lambda x : getsummary(x))
```

```
dfcontent2['summary'] = dfcontent2["text"].apply(lambda x : getsummary(x))
```

- Now union the dataframes

```
dffinal = pd.concat([dfcontent, dfcontent1])
```

```
dffinal1 = pd.concat([dffinal, dfcontent2])
```

- remove special characters

```
### Define function
def remove_special_characters(df_column,bad_characters_list):
    clean_df_column = df_column
    for bad_char in bad_characters_list:
        clean_df_column = clean_df_column.str.replace(bad_char,' ')
        print("row changes in column " + str(df_column.name) + " after removing character " + str(bad_char) + ": " ,sum(df_column!=clean_df_column))
    clean_df_column = clean_df_column.str.title()
    return clean_df_column
```

- remove utf

```
def remote_non_utf8(name):
     return re.sub(r'[^\x00-\x7f]',r' ',name)
        
dffinal1['summary'] = dffinal1['summary'].apply(remote_non_utf8)
```

- Remove bad characters

```
### Run function
bad_chars_lst = ["*","!","?", "(", ")", "-", "_", ",", "\n", "\\r\\n", "\r"]
dffinal1['summary'] = remove_special_characters(dffinal1['summary'],bad_chars_lst)
dffinal1['text'] = remove_special_characters(dffinal1['text'],bad_chars_lst)
display(dffinal1[["summary"]].head(20))
```

- Clear new line

```
dffinal1['summary'].replace({ r'\A\s+|\s+\Z': '', '\n' : ' '}, regex=True, inplace=True)
```

- Remove consecutive spaces

```
### Define function
def remove_consecutive_spaces(df_column):
    clean_df_column = df_column.replace('\s\s+', ' ', regex=True)
    print("row changes in column " + str(df_column.name) +": " ,sum(df_column!=clean_df_column))
    return clean_df_column

### Run function
dffinal1['text'] = remove_consecutive_spaces(dffinal1['text'])
dffinal1['summary'] = remove_consecutive_spaces(dffinal1['summary'])
```

- finally save the dataframe as csv file

```
dffinal1.to_csv('name.csv', header=True, index=False)
```