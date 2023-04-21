# Process Embedding for large rows of data with single api within limit

## How to process embedding for large rows of data with single api within limit

- This is a sample notebook to process embedding for large rows of data with single api within limit
- Ada 2 will have a limit of 30 rows within few seconds to process.
- This sample process 30 rows at a time and wait for 7 seconds to avoid throttling
- Some libraries retry but we do loose data
- Wanted to make sure all rows are processed for embedding in pandas dataframe


## Code

- Import libraries

```
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
```

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoairesroucename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxxxxxxxxxxxxx"
OpenAiKey = "xxxxxxxxxxxxxxxxxxxxxxx"
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

- now read the data

```
df = pd.read_json('accindex.json')
```

```
display(df.head())
```

- Calculate the token

```
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df['n_tokens'] = df["content"].apply(lambda x: len(tokenizer.encode(x)))
#df = df[df.n_tokens<2000]
len(df)
```

- Create a function to process embedding

```
import pandas as pd
import tiktoken

from openai.embeddings_utils import get_embedding

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   #print(text)
   embedding = openai.Embedding.create(input=[text], deployment_id="text-embedding-ada-002")
   return embedding['data'][0]['embedding']
```

```
df2 = df.copy()
```

- calculate the size

```
chunksize = 30
start = 0
total = len(df)
end = total
print(end)
```

- Copy the dataframe

```
df3 = df.copy()
```

- chunk the dataframe and process embedding

```
import numpy as np
import time

df5 = pd.DataFrame()
df_list = []

for i in range(start, len(df3), chunksize) :
    #display(df1.iloc[i:chunksize])
    df5 = pd.DataFrame()
    df4 = df3.iloc[int(i):int(chunksize + i)].copy()
    print(str(i) + " " + str(len(df3)))    
    try:
        #processdf(client, df3)
        df4['ada_embedding'] = df4["content"].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        #df5.concat(df4)
        df_list.append(df4)
        #df3.to_csv('eadocembed.csv', mode='a', index=False, header=False)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    time.sleep(7)
    i = i + chunksize
```

- now concat the dataframe

```
final_df = pd.concat(df_list)
```

- count the rows

```
final_df.count()
```

- Check the null

```
final_df["ada_embedding"].isna().sum()
```

- create the new columns for cognitive search

```
final_df["titleVector"] = final_df["ada_embedding"]
final_df["contentVector"] = final_df["ada_embedding"]
final_df["category"] = "web"
final_df["@search.action"] = "upload"
final_df['id'] = final_df['id'].apply(str)
```

- Select only columns

```
dfj1 = final_df[["id","title", "content", "category", "titleVector", "contentVector", "@search.action"]]
```

```
dfj1.dtypes
```

- now save the dataframe to json

```
final_df.to_csv('eadocembed.csv', header=True, index=False, mode='w')
```

```
import requests
``

- set authentication for cognitive search

```
my_headers = {"Content-Type" : "application/json", "api-key" : "xxxxxxxxxxxxxxx"}
```

```
df2 = dfj1.copy()
```

```
len(df2)
```

- send the embedding to cognitive search

```

url = 'https://vecacctest.search.windows.net/indexes/vecaccindex/docs/index?api-version=2023-07-01-Preview'
headers = {'api-key' : 'lMoMhGCFH4mAqBVqHsLysvNFJSEpnw5Z3EttlmVIdhAzSeCByQbO', 'Content-Type' : 'application/json'}

for id, row in df2.iterrows():
    payload = {
      "value": [
        {
          "id": str(id),
          "title": row['title'],
          "content": row['content'], #Example InSpec Profile For Azure This example shows the implementation of an InSpec profile for Azure. See https:\\/\\/github.com\\/inspec\\/inspec-azure for details on how to configure credentials for your subscription.\\nCreate a profile $ inspec init profile --platform azure my-profile \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500 InSpec Code Generator \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500 Creating new profile at \\/Users\\/spaterson\\/my-profile \\u2022 Creating directory libraries \\u2022 Creating file README.md \\u2022 Creating directory controls \\u2022 Creating file controls\\/example.rb \\u2022 Creating file inspec.yml \\u2022 Creating file libraries\\/.",
          "category": "web",
          "titleVector": row['titleVector'],
          "contentVector": row['contentVector'],
          "@search.action": "upload"
        }
      ]

    }
    #print(payload)
    response = requests.request('POST', url, headers=headers, json=payload)
    print(response.json())
    #break
```

- Output in cog search

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/cogsearch1.jpg "Output Episodes")