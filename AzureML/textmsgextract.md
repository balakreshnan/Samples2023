# Azure Open AI extract entities from text message using Azure Machine Learning

## Using Azure Machine learning and Open AI to parse Text messages information

## Pre-requisites

- Azure Storage
- Azure Machine Learning
- Azure Open AI service
- Sample data from Kaggle

## Code

- Go to Azure Machine learning serivce
- Start the compute instance
- Open Jupyter notebook
- Select python3.10 kernel with azure ml sdk
- Install packages

```
pip install openai
```

- now tiktoken

```
pip install tiktoken
```

- Configure open ai configuration

```
#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxx"
```

- Set the environment variables for openao

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://aoainame.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


- Now setup the model and load libraries

```
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-35-turbo"
GPT_DEPLOYNAME = "gpt-35-turbo"
```

- load the data

```
import pandas as pd
```

- load the data

```
df = pd.read_csv('textmessage1.csv')
```

- Function to process text message
- formatting for chatgpt 3.5 turbo

```
def getcategory(mystring):
    query = mystring + " classify the above text either as urgent police, urgent medical, drug related, gun shot, fire, burglary, others. output only category. Classify:"
    #print("Query: ", query)
    response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    engine=GPT_DEPLOYNAME,
    model=GPT_MODEL,
    temperature=0,
    )
    return response['choices'][0]['message']['content']
```

- now process the dataframe

```
df['category'] = df["textmessage"].apply(lambda x : getcategory(x))
```

- now display the dataframe

```
df.head()
```

- Save the dataframe as csv file for future use

```
df.to_csv('textmessage2.csv', index=False)
```