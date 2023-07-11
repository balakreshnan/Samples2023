# Talk to structured data using yolopandas and Azure Open AI GPT-3.5-turbo

## Introduction

- ability to talk to structured data
- using yolopandas and Azure Open AI GPT-3.5-turbo

## code

- install yolopandas
- import open ai and configure for azure

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "xxxxx"
```

- Set environment variables

```
import os
os.environ["OPENAI_API_KEY"] = "xxxxxx"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://aoainame.openai.azure.com/"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"

os.environ["AZURE_OPENAI_KEY"] = "xxxxxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoainame.openai.azure.com/"
```

- Test Azure open ai response

```
import os
import requests
import json
import openai

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

deployment_name='davinci' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=10)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(start_phrase+text)
```

- import chatapi

```
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
```

- Configure LLM to use with yolopandas

```
BASE_URL = "https://aoainame.openai.azure.com/"
API_KEY = "xxxxx"
DEPLOYMENT_NAME = "chatgpt"
llm1 = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)
```

- Set up yolopandas

```
import yolopandas
yolopandas.set_llm(llm1)
```

- load the data frame

```
import mltable

tbl = mltable.load("./data")
df = tbl.to_pandas_dataframe()
```

```
df.llm.query("how many rows are there?", yolo=True)
```

```
df.llm.query("create a seaborn charts for total count of protocols?", yolo=True)
```

```
df.llm.query("Create seaborn charts of all destination address?",yolo=True)
```

```
df.llm.query("create seaborn charts for total count of protocols?", yolo=True)
```