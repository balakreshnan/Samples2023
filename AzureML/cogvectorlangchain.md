# Using Langchain vector store using Azure Cognitive Search

## Overview

- Use Azure Cognitive Search to store and retrieve vectors from a vector store
- using langchain
- use open ai embeddings to create a vector

## Prerequisites

- Azure Account
- Azure Cognitive Search
- Azure Open AI
- Azure Storage Account

## Code

- install langchain

```
pip install --upgrade langchain
pip install --index-url=https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/ azure-search-documents==11.4.0a20230509004
pip install azure-identity
pip install pypdf
```

- import libraries

```
import os, json
import openai
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
```

- Configure openai

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "xxxxx"
```

```
import os
os.environ["OPENAI_API_KEY"] = "xxxx"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://aoainame.openai.azure.com/"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"

os.environ["AZURE_OPENAI_KEY"] = "xxxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoainame.openai.azure.com/"
```

- Set up embedding models

```
model: str = "text-embedding-ada-002"
```

- Configure cogntive search configuration

```
vector_store_address: str = "https://cogsvcname.search.windows.net"
vector_store_password: str = "xxxxxxxxxxxxxxxx"
index_name: str = "langchain-vector-demo"
```

- Setup embedding and search

```
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model, chunk_size=1)
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)
```

- load the padf

```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdf1/Blueprint-for-an-AI-Bill-of-Rights.pdf")
pages = loader.load_and_split()
```

```
from langchain.text_splitter import CharacterTextSplitter
```

- Create a vector store

```

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vector_store.add_documents(documents=docs)
```

- Search for a vector

```
# Perform a similarity search
docs = vector_store.similarity_search(
    query="what is requirement for responsible ai?",
    k=3,
    search_type="similarity",
)
print(docs[0].page_content)
```

```
# Perform a hybrid search
docs = vector_store.similarity_search(
    query="what is requirement for responsible ai?", k=3
)
print(docs[0].page_content)
```

```