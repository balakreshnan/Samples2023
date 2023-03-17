# Azure Open AI and vector search with FAISS

## Pre-requisites

- Azure Storage
- Azure Machine Learning
- Azure Machine learning
- FAISS

## Code

- Go to Azure Machine learning serivce
- Open jupyter on the compute instance
- Install packages

```
pip install faiss-cpu
```

```
pip install --upgrade openai
```

- Configure open ai

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoairesource.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxx"
```

- Test chat gpt

```
response = openai.Completion.create(
  engine="gpt-35-turbo",
  prompt="how are you?",
  temperature=0.7,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
```

- Set up environment variables

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://aoairesoruce.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxx"
```

- Test chatgpt

```
import openai

response = openai.Completion.create(
    engine="davinci003",
    prompt="This is a test",
    max_tokens=5
)
```

- Load document

```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/Blueprint-for-an-AI-Bill-of-Rights.pdf")
pages = loader.load_and_split()
```

- load libraries

```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
```

- Create embeddings

```
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(openai_api_key="xxxxxxx", chunk_size=1500)
print(embeddings.json)
```

- Import FAISS

```
from langchain.vectorstores import FAISS

#db = FAISS.from_documents(docs, embeddings)
print(embeddings.embed_query)
```

- create texts for FAISS

```
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#docs = text_splitter.split_documents(documents)
docs = str(text_splitter.split_documents(documents))

embeddings = OpenAIEmbeddings(openai_api_key="xxxxxx", chunk_size=1000)
print(embeddings.json)
```

- Create FAISS

```
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

embeddings = OpenAIEmbeddings()
faiss = FAISS.from_texts(docs, embeddings)
```

- Search query

```
question = "Data privacy"
```

- Setup

```
docs_db = faiss.similarity_search(question)
```

- imports

```
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
```

- setup qna

```
llm = AzureOpenAI(deployment_name="davinci003", model_name="text-davinci-003", temperature=0.5, max_tokens=500) 
chain = load_qa_chain(llm, chain_type="refine")
```

- send to openai

```
response = chain({"input_documents": docs_db, "question": question, "language": "English", "existing_answer" : ""}, return_only_outputs=True)
```

- print response

```
print(response['output_text'])
```
