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

- Configure LLM

```
import os
#from dotenv import load_dotenv
#from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
```

- LLM Config

```
from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

deployment_name = "davinci"

#llm = OpenAI(deployment_id="davinci",temperature=0)
#llm = AzureOpenAI(deployment_name="davinci", model_name="text-davinci-003", openai_api_type="azure",)
llm = AzureChatOpenAI(deployment_name="chatgpt",
                      model_name="gpt-35-turbo",
                      openai_api_base="https://aoainame.openai.azure.com/",
                      openai_api_version="2023-05-15",
                      openai_api_key="xxxxxxxxxxxxxxxx",
                      openai_api_type="azure")
```

- Setup LLM

```
import yolopandas
yolopandas.set_llm(llm)
```

- Load the dataset

```
df = pd.read_csv('Datafor10 Clients_Cols_withVlookup_Sep_Mar.csv')
```

- now run queries

```
df.llm.query("Show me total number of rows?", yolo=True)
```

- output 

```
17808
```

- Show all the columns configuration

```
pd.set_option('display.max_columns', None)
```

- more queries

```
df.llm.query("Show me total by clients?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas1.jpg "Architecture")

```
df.llm.query("Show me total by clients by Period?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas2.jpg "Architecture")

- query

```
df.llm.query("Show me total by clients and contract by Period?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas3.jpg "Architecture")

- by cost

```
df.llm.query("Show me total hours by clients and contract by Period?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas4.jpg "Architecture")

- By Revenue

```
df.llm.query("Show me total Actual Revenue by clients and contract by Period?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas5.jpg "Architecture")

- Now charting


```
df.llm.query("Chart me total client by Revenue?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas6.jpg "Architecture")

- By Cost

```
df.llm.query("Chart me total client by Cost?", yolo=True)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/yolopandas7.jpg "Architecture")