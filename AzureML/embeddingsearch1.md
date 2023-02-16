# Efficient Embedding Search for Natural Language Queries

## Introduction

- use open ai and create embedding
- Search columns based on embeddings
- Demo application
- Restrict 40 rows to azure open ai API

## install packages

```
pip install tiktoken
```

```
pip install openai
```

## Code

- Create encoding for gpt2 to get tokens

```
encoding = tiktoken.get_encoding("gpt2")
```

- Read the data

```
import pandas as pd

df = pd.read_json('accindex.json')
```

- model configuration

```
model = 'text-similarity-ada-001'
```

- Now configure open ai services

```
import os
import openai
import pandas as pd
openai.api_type = "azure"
openai.api_base = "https://resoufcename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "key"
```

- convert data to dataframe

```
dfcontent = pd.DataFrame(df["content"])
dfcontent = pd.DataFrame(df)
```

- Normilize the data

```
# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

dfcontent['content'] = dfcontent["content"].apply(lambda x : normalize_text(x))
```

- now to get tokens

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

- Calculate tokens

```
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
dfcontent['n_tokens'] = dfcontent["content"].apply(lambda x: len(tokenizer.encode(x)))
dfcontent = dfcontent[dfcontent.n_tokens<2000]
len(dfcontent)
```

- now filter the dataset to 40 rows

```
dfcontent1 = dfcontent.iloc[:40].copy()
```

- Create Embeddings

```
dfcontent1['curie_search'] = dfcontent1["content"].apply(lambda x : get_embedding(x, engine = 'text-similarity-ada-001'))
```

- Now create a function to search the embeddings

```
# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-similarity-ada-001"
    )
    df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        display(res)
    return res


res = search_docs(dfcontent1, "how can i create Google FileStore", top_n=4)
#res = search_docs(dfcontent1, "how to build inspec profile", top_n=4)
```

- Display results for one row

```
res["content"][6]
```
