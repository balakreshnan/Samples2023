# Azure Open AI Summarize Meeting notes

## Overview

- Summarize Meeting converstation transcript
- Load Text data memory
- Clean the data
- Load the data using Langchain
- Using Azure Machine learning
- Upload meeting transcript into a folder

## Code

- install libraries

```
pip install pdfreader
pip install langchain
pip install unstructured
pip install tiktoken
pip install faiss-cpu
```

- load libraries

```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
```

- Load environment variable for open ai configuration and keys
- Load the Azure open ai endpoint

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://resourcename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxxxxxx"
```

```
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

llm = OpenAI(engine="davinci",temperature=0)

text_splitter = CharacterTextSplitter()
```

- Load the text data

```
from langchain.document_loaders import TextLoader
loader = TextLoader('./meetingtranscript.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(chunk_size=1)
```

- load the file

```
with open("./meetingtranscript.txt") as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)
```

- load langchain summarizer

```
from langchain.chains.summarize import load_summarize_chain
```

- now configure the ChatOpenAI

```
from langchain.chat_models import ChatOpenAI
llm=ChatOpenAI(temperature=0.7, engine="chatgpt", max_tokens=300)
```

- To sumamrize the entire text

```
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

- Now to prompt engineering to generate Action items


```
prompt_template = """Extract a Action items for follow-up:


{text}


summarize tasks:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
chain.run(docs)
```