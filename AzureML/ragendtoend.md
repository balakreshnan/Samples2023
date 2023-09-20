# Talk to your data using Azure Open AI - RAG pattern architecture in Azure Machine Learning

## Talk to your data

### Pre-requisites

- Azure subscription
- Azure Storage account
- Azure Machine Learning workspace
- Azure Cognitive Search service
- Azure Open AI

### Introduction

- Upload your own data into Azure Blob Storage
- For this tutorial, i am using open source data set
- Public data exported from linken profiles
- data in pdf format
- Once you have uploaded the pdf to blob storage
- I am using compute instance to read pdf, download and create Cognitive search vector index
- All code can run in notebook using notebok
- create a environment file and load all the keys

## Code

- Include libraries

```
import re
import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
import requests

from collections import OrderedDict
import base64

import docx2txt
import tiktoken
import html
import time
from pypdf import PdfReader, PdfWriter
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
```

```
import os
import json
import time
import requests
import random
from collections import OrderedDict
import urllib.request
from tqdm import tqdm
import langchain
```

```
import os
import json
import time
import requests
import random
from collections import OrderedDict
import urllib.request
from tqdm import tqdm
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain



from IPython.display import Markdown, HTML, display  

from dotenv import load_dotenv
load_dotenv("env2.env")

def printmd(string):
    display(Markdown(string))
    
#os.makedirs("data/books/",exist_ok=True)
    

BLOB_CONTAINER_NAME = "acccsuite"
BASE_CONTAINER_URL = "https://synpasedlstore.blob.core.windows.net/" + BLOB_CONTAINER_NAME + "/"
#LOCAL_FOLDER = "./data/books/"

MODEL = "gpt-35-turbo-16k" # options: gpt-35-turbo, gpt-35-turbo-16k, gpt-4, gpt-4-32k

#os.makedirs(LOCAL_FOLDER,exist_ok=True)
```

- Load the environment file

```
from dotenv import dotenv_values
# specify the name of the .env file name 
env_name = "env2.env" # change to use your own .env file
config = dotenv_values(env_name)
```

- Set Environment variables

```
# Set the ENV variables that Langchain needs to connect to Azure OpenAI
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
os.environ["OPENAI_API_TYPE"] = "azure"
```

- Create function to parse PDF using Form Recognizer to split page

```
def parse_pdf(file, form_recognizer=False, formrecognizer_endpoint=None, formrecognizerkey=None, model="prebuilt-document", from_url=False, verbose=False):
    """Parses PDFs using PyPDF or Azure Document Intelligence SDK (former Azure Form Recognizer)"""
    offset = 0
    page_map = []
    if not form_recognizer:
        if verbose: print(f"Extracting text using PyPDF")
        reader = PdfReader(file)
        pages = reader.pages
        for page_num, p in enumerate(pages):
            page_text = p.extract_text()
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)
    else:
        if verbose: print(f"Extracting text using Azure Document Intelligence")
        credential = AzureKeyCredential(os.environ["FORM_RECOGNIZER_KEY"])
        form_recognizer_client = DocumentAnalysisClient(endpoint=os.environ["FORM_RECOGNIZER_ENDPOINT"], credential=credential)
        
        if not from_url:
            with open(file, "rb") as filename:
                poller = form_recognizer_client.begin_analyze_document(model, document = filename)
        else:
            poller = form_recognizer_client.begin_analyze_document_from_url(model, document_url = file)
            
        form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif not table_id in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)

    return page_map
```

- Create function to convert PDF to HTML

```
def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html
```

- text to base 64

```
def text_to_base64(text):
    # Convert text to bytes using UTF-8 encoding
    bytes_data = text.encode('utf-8')

    # Perform Base64 encoding
    base64_encoded = base64.b64encode(bytes_data)

    # Convert the result back to a UTF-8 string representation
    base64_text = base64_encoded.decode('utf-8')

    return base64_text
```

- Create index name

```
profile_index_name = "cogsrch-index-profile-vector"
```

- Set headers for cognitive search

```
### Create Azure Search Vector-based Index
# Setup the Payloads header
headers = {'Content-Type': 'application/json','api-key': os.environ['AZURE_SEARCH_KEY']}
params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}
```

- configure the index schema

```
index_payload = {
    "name": profile_index_name,
    "fields": [
        {"name": "id", "type": "Edm.String", "key": "true", "filterable": "true" },
        {"name": "title","type": "Edm.String","searchable": "true","retrievable": "true"},
        {"name": "chunk","type": "Edm.String","searchable": "true","retrievable": "true"},
        {"name": "chunkVector","type": "Collection(Edm.Single)","searchable": "true","retrievable": "true","dimensions": 1536,"vectorSearchConfiguration": "vectorConfig"},
        {"name": "name", "type": "Edm.String", "searchable": "true", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
        {"name": "location", "type": "Edm.String", "searchable": "false", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
        {"name": "page_num","type": "Edm.Int32","searchable": "false","retrievable": "true"},
        
    ],
    "vectorSearch": {
        "algorithmConfigurations": [
            {
                "name": "vectorConfig",
                "kind": "hnsw"
            }
        ]
    },
    "semantic": {
        "configurations": [
            {
                "name": "my-semantic-config",
                "prioritizedFields": {
                    "titleField": {
                        "fieldName": "title"
                    },
                    "prioritizedContentFields": [
                        {
                            "fieldName": "chunk"
                        }
                    ],
                    "prioritizedKeywordsFields": []
                }
            }
        ]
    }
}

r = requests.put(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + profile_index_name,
                 data=json.dumps(index_payload), headers=headers, params=params)
print(r.status_code)
print(r.ok)
```

- only use this for troubleshooting

```
# Uncomment to debug errors
# r.text
```

- Now Loop all pdf and save locally to process

```
from typing import Container
#from azure.cosmosdb.table.tableservice import TableService,ListGenerator
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from azure.storage.blob import ResourceTypes, AccountSasPermissions
from azure.storage.blob import generate_account_sas    
from datetime import *

today = str(datetime.now().date())
print(today)

#================================ SOURCE ===============================
# Source Client
connection_string = 'xxxxxxxxxxxxxxxxxxxxxx' # The connection string for the source container
account_key = 'xxxxxxxxxxxxxxxxx' # The account key for the source container
# source_container_name = 'newblob' # Name of container which has blob to be copied
#connection_string = config["DOCUMENT_AZURE_STORAGE_CONNECTION_STRING"]
#account_key= config["DOCUMENT_AZURE_STORAGE_CONNECTION_KEY"]

# Create client
client = BlobServiceClient.from_connection_string(connection_string) 

client = BlobServiceClient.from_connection_string(connection_string)
all_containers = client.list_containers(include_metadata=True)

for container in all_containers:
    #print("==========================")
    print(container['name'], container['metadata'])
    if container['name'] == 'acccsuite':
        container_client = client.get_container_client(container.name)
        blobs_list = container_client.list_blobs()
        for blob in blobs_list:
            target_blob_name = blob.name
            print(target_blob_name)
            blob_client = client.get_blob_client(container=container['name'], blob=blob.name)
            with open(file=os.path.join(r'', blob.name), mode="wb+") as sample_blob:
                download_stream = blob_client.download_blob(max_concurrency=2)
                destfile = os.path.join(r'/data/profiles', blob.name)
                sample_blob.write(download_stream.readall())
                #with open(destfile, "w") as file:
                #    file.write(download_stream.readall())
                #    file.close()

```

- Move the file from current folder to data/profiles

```
!mv *.pdf data/profiles
```

- Now loop the local pdf and add it to list

```
import os
# assign directory
directory = 'data/profiles'
filepdfs= []
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        #print(f)
        filepdfs.append(f)
```

- Process the above list and split pages

```
pdf_pages_map = dict()
for pdf in filepdfs:
    print("Extracting Text from",pdf,"...")
    
    if not '.amlignore' in pdf:

        # Capture the start time
        start_time = time.time()
        
        # Parse the PDF
        pdf_path = pdf
        pdf_map = parse_pdf(file=pdf_path, form_recognizer=True, verbose=True)
        pdf_pages_map[pdf]= pdf_map
        
        # Capture the end time and Calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Parsing took: {elapsed_time:.6f} seconds")
        print(f"{pdf} contained {len(pdf_map)} pages\n")
```

```
print(f"Total contained {len(pdf_pages_map)} pages\n")
```

- now prep for creating embedding

```
import openai

openai.api_type = "azure"
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
openai.api_version = "2023-05-15"
```

- Embedding

```
import openai

def getmebedding(text):
    response = openai.Embedding.create(
    input=text,
    engine="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']
```

- Loop the pages and create index

```
for pdfname,pdfmap in pdf_pages_map.items():
    print("Uploading chunks from",pdfname)
    for page in tqdm(pdfmap):
        try:
            page_num = page[0] + 1
            content = page[2]
            #print('Content::', content)
            head_tail = os.path.split(pdfname)
            book_url = BASE_CONTAINER_URL + head_tail[1]
            upload_payload = {
                "value": [
                    {
                        "id": text_to_base64(pdfname + str(page_num)),
                        "title": f"{pdfname}_page_{str(page_num)}",
                        "chunk": content,
                        "chunkVector": getmebedding(content if content!="" else "-------"),
                        "name": pdfname,
                        "location": book_url,
                        "page_num": page_num,
                        "@search.action": "upload"
                    },
                ]
            }

            r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + profile_index_name + "/docs/index",
                                 data=json.dumps(upload_payload), headers=headers, params=params)
            if r.status_code != 200:
                print(r.status_code)
                print(r.text)
        except Exception as e:
            print("Exception:",e)
            print(content)
            continue
```

- Now let's create a client search demo

```
QUESTION = "List top 10 leaders for technology strategy with details of their experience?"
```

- Function to get the answer

```
def get_search_results(query: str, indexes: list, 
                       k: int = 5,
                       reranker_threshold: int = 1,
                       sas_token: str = "",
                       vector_search: bool = False,
                       similarity_k: int = 3, 
                       query_vector: list = []) -> List[dict]:
    
    headers = {'Content-Type': 'application/json','api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    agg_search_results = dict()
    
    for index in indexes:
        search_payload = {
            "search": query,
            "queryType": "semantic",
            "semanticConfiguration": "my-semantic-config",
            "count": "true",
            "speller": "lexicon",
            "queryLanguage": "en-us",
            "captions": "extractive",
            "answers": "extractive",
            "top": k
        }
        if vector_search:
            search_payload["vectors"]= [{"value": query_vector, "fields": "chunkVector","k": k}]
            search_payload["select"]= "id, title, chunk, name, location"
        else:
            search_payload["select"]= "id, title, chunks, language, name, location, vectorized"
        

        resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                         data=json.dumps(search_payload), headers=headers, params=params)

        search_results = resp.json()
        agg_search_results[index] = search_results
    
    content = dict()
    ordered_content = OrderedDict()
    
    for index,search_results in agg_search_results.items():
        for result in search_results['value']:
            if result['@search.rerankerScore'] > reranker_threshold: # Show results that are at least N% of the max possible score=4
                content[result['id']]={
                                        "title": result['title'], 
                                        "name": result['name'], 
                                        "location": result['location'] + sas_token if result['location'] else "",
                                        "caption": result['@search.captions'][0]['text'],
                                        "index": index
                                    }
                if vector_search:
                    content[result['id']]["chunk"]= result['chunk']
                    content[result['id']]["score"]= result['@search.score'] # Uses the Hybrid RRF score
              
                else:
                    content[result['id']]["chunks"]= result['chunks']
                    content[result['id']]["language"]= result['language']
                    content[result['id']]["score"]= result['@search.rerankerScore'] # Uses the reranker score
                    content[result['id']]["vectorized"]= result['vectorized']
                
    # After results have been filtered, sort and add the top k to the ordered_content
    if vector_search:
        topk = similarity_k
    else:
        topk = k*len(indexes)
        
    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding 5 results
            break

    return ordered_content
```

- now search for results

```
vector_indexes = [profile_index_name]

ordered_results = get_search_results(QUESTION, vector_indexes, 
                                        k=10,
                                        reranker_threshold=1,
                                        vector_search=True, 
                                        similarity_k=10,
                                        query_vector = getmebedding(QUESTION)
                                        )
```

- Assign LLM

```
COMPLETION_TOKENS = 1000
llm = AzureChatOpenAI(deployment_name="gpt-35-turbo-16k", temperature=0.5, max_tokens=COMPLETION_TOKENS, openai_api_version="2023-05-15")
```

- Now loop the results and get the answer

```
top_docs = []
for key,value in ordered_results.items():
    location = value["location"] if value["location"] is not None else ""
    top_docs.append(Document(page_content=value["chunk"], metadata={"source": location+os.environ['BLOB_SAS_TOKEN']}))
        
print("Number of chunks:",len(top_docs))
```

- utility function for parsing results

```
# Returns the num of tokens used on a string
def num_tokens_from_string(string: str) -> int:
    encoding_name ='cl100k_base'
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Returning the toekn limit based on model selection
def model_tokens_limit(model: str) -> int:
    """Returns the number of tokens limits in a text model."""
    if model == "gpt-35-turbo":
        token_limit = 4096
    elif model == "gpt-4":
        token_limit = 8192
    elif model == "gpt-35-turbo-16k":
        token_limit = 16384
    elif model == "gpt-4-32k":
        token_limit = 32768
    else:
        token_limit = 4096
    return token_limit

# Returns num of toknes used on a list of Documents objects
def num_tokens_from_docs(docs: List[Document]) -> int:
    num_tokens = 0
    for i in range(len(docs)):
        num_tokens += num_tokens_from_string(docs[i].page_content)
    return num_tokens
```

- Now test the prompt

```
from langchain.prompts import PromptTemplate


COMBINE_QUESTION_PROMPT_TEMPLATE = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text in {language}.
{context}
Question: {question}
Relevant text, if any, in {language}:"""

COMBINE_QUESTION_PROMPT = PromptTemplate(
    template=COMBINE_QUESTION_PROMPT_TEMPLATE, input_variables=["context", "question", "language"]
)


COMBINE_PROMPT_TEMPLATE = """
# Instructions:
- Given the following extracted parts from one or multiple documents, and a question, create a final answer with references. 
- You can only provide numerical references to documents, using this html format: `<sup><a href="url?query_parameters" target="_blank">[number]</a></sup>`.
- The reference must be from the `Source:` section of the extracted part. You are not to make a reference from the content, only from the `Source:` of the extract parts.
- Reference (source) document's url can include query parameters, for example: "https://example.com/search?query=apple&category=fruits&sort=asc&page=1". On these cases, **you must** include que query references on the document url, using this html format: <sup><a href="url?query_parameters" target="_blank">[number]</a></sup>.
- **You can only answer the question from information contained in the extracted parts below**, DO NOT use your prior knowledge.
- Never provide an answer without references.
- If you don't know the answer, just say that you don't know. Don't try to make up an answer.
- Respond in {language}.

=========
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN {language}:"""


COMBINE_PROMPT = PromptTemplate(
    template=COMBINE_PROMPT_TEMPLATE, input_variables=["summaries", "question", "language"]
)
```

- Now calculate the token

```
# Calculate number of tokens of our docs
if(len(top_docs)>0):
    tokens_limit = model_tokens_limit(MODEL) # this is a custom function we created in common/utils.py
    prompt_tokens = num_tokens_from_string(COMBINE_PROMPT_TEMPLATE) # this is a custom function we created in common/utils.py
    context_tokens = num_tokens_from_docs(top_docs) # this is a custom function we created in common/utils.py
    
    requested_tokens = prompt_tokens + context_tokens + COMPLETION_TOKENS
    
    chain_type = "map_reduce" if requested_tokens > 0.9 * tokens_limit else "stuff"  
    
    print("System prompt token count:",prompt_tokens)
    print("Max Completion Token count:", COMPLETION_TOKENS)
    print("Combined docs (context) token count:",context_tokens)
    print("--------")
    print("Requested token count:",requested_tokens)
    print("Token limit for", MODEL, ":", tokens_limit)
    print("Chain Type selected:", chain_type)
        
else:
    print("NO RESULTS FROM AZURE SEARCH")
```

- Configure the chain

```
if chain_type == "stuff":
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                       prompt=COMBINE_PROMPT)
elif chain_type == "map_reduce":
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                       question_prompt=COMBINE_QUESTION_PROMPT,
                                       combine_prompt=COMBINE_PROMPT,
                                       return_intermediate_steps=True)
```

- Now get the answer

```
%%time
# Try with other language as well
response = chain({"input_documents": top_docs, "question": QUESTION, "language": "English"})
```

- display the sumamrized text

```
display(Markdown(response['output_text']))
```