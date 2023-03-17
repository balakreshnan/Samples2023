# Azure Open AI process survey forms and Extract information

## Using Azure Machine learning and Open AI to parse Survey data

## Pre-requisites

- Azure Storage
- Azure Machine Learning
- Azure Open AI service
- Sample data from Kaggle
- Sample data uploaded in survey/ folder in github repo

## Code

- Go to Azure Machine learning serivce
- Start the compute instance
- Open Jupyter notebook
- Install packages

```
pip install openai
```

- Import libraries

```
import pandas as pd
```

- Load the survey data

```
df = pd.read_csv('survey/freeformResponses.csv')
```

- Set pandas options

```
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

- display the data

```
df.head()
```

- diplay columns

```
df.columns
```

- Display unique values

```
df.nunique()
```

- Now we can filter only data that has values
- Get rid of empty rows
- I am only picking one question and answer to process here to show

```
df["InterestingProblemFreeForm"].unique()
```

- Now only we take one column to process - InterestingProblemFreeForm

```
df1 = df["InterestingProblemFreeForm"].dropna()
#df1 = df1.set_axis(['index','InterestingProblemFreeForm'])
df = df.reset_index()
print(df)
```

- Display the dataset to only one column

```
df1.head()
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/images/survey1.jpg "Output Episodes")

- now do a count

```
df1.count()
```

- check the type of df1

```
type(df1)
```

- Here is the code to convert pandas series to dataframe for us to process

```
df2 = df1.to_frame()
```

- display columns

```
df2.columns
```

- Now bring open ai configuration


```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoairesource.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxxxx"
```

- Create tokens

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

```
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df2['n_tokens'] = df2['InterestingProblemFreeForm'].apply(lambda x: len(tokenizer.encode(x)))
#df2 = df2[df2.n_tokens<2000]
len(df2)
```

- Now function to extract information using open ai

```
def getsummary(mystring):
    bad_chars_lst = ["*","!","?", "(", ")", "-", "_", ",", "\n", "\\r\\n", "\r"]
    response = openai.Completion.create(
    engine="davinci003",
    prompt= 'Extract Sentiment, Product, Context, Category in a comma separated key value format from below text ' + mystring,
    temperature=0.9,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1
    )
    for bad_char in bad_chars_lst:
        clean_df_column = response.choices[0].text.strip().rstrip().replace(bad_char,' ')
    return clean_df_column
```

- Calculate total

```
total = df2.count()
```

- Configure chunks to process


```
chunksize = 20
start = 0
end = total
print(end)
```

- create a blank dataframe for results

```
dffinal = df2.iloc[0:0]
```

```
dffinal.columns
```

- Display the dataframe

```
display(df2)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/images/survey2.jpg "Output Episodes")

- Process dataframe 20 at a time ,we have about 4000 rows

```
for i in range(start, len(df1), chunksize) :
    #display(df1.iloc[i:chunksize])
    df2 = df2.iloc[int(i):int(chunksize + i)].copy()
    
    df2['extract'] = df2["InterestingProblemFreeForm"].apply(lambda x : getsummary(x))
    #display(df2)
    df2.to_csv('datawithextract1.csv', mode='a', index=False, header=False)
    #pd.concat([dffinal, df2[["text", "n_tokens"]]])
    #dffinal.append(df2, ignore_index=True)
    #print(dffinal.count())
    print(i)
    
    i = i + chunksize
    #df2.drop(axis=0, inplace=True)
    #print(i)
```

- Now read the file saved

```
df3 = pd.read_csv('datawithextract1.csv')
```

- Add columns to the dataframe

```
df3.columns = ['InterestingProblemFreeForm', 'n_tokens', 'extract']
pd.set_option('display.max_colwidth', None)
```

- Display the data

```
display(df3)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/images/survey3.jpg "Output Episodes")