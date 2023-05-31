# QnA with large RFP document as pdf

## Overview

- Ability to read large pdf
- Ask any question about the document
- Get the answer from the document

## Code

- First import
- install necessary packages

```
pip install --upgrade openai
pip install pypdf
pip install tabula-py
```

- Import necessary packages

```
import os
import requests
import json
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
```

- Set the openai api key

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoai.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "xxxxxxxxxxxxx"
```

- Set environment variable

```
import os
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://aoai.openai.azure.com/"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"

os.environ["AZURE_OPENAI_KEY"] = "xxxxxxxxxxxxxxxxxxxxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aoai.openai.azure.com/"
```

- set the deploy

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

- Set the pdf file

```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("RFP.pdf")
pages = loader.load_and_split()
```

- create the index

```
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(chunk_size=1))
```

```
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
```

- create the chain

```
docs = faiss_index.similarity_search("when is the first and second cutover for PPM?", k=5)
chain = load_qa_with_sources_chain(OpenAI(deployment_id= deployment_name, temperature=0), chain_type="stuff")
query = "when is the first and second cutover for PPM?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
```

- Get the answer

```
query = "Is there a parallel requirement? If yes, what is the expectation?"
template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in English.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN English:"""
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

chain = load_qa_with_sources_chain(OpenAI(deployment_id=deployment_name,temperature=0), chain_type="stuff", prompt=PROMPT)
#query = "What did the president say about Justice Breyer"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
```
