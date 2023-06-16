# Azure Cognitive Search with cosmos db with Natural Language Processing

## Able to ask natural language query

## Pre-requisites

- Azure Account
- Azure Cosmos db
- Azure cognitive search
- Azure machine learning service
- Azure Open AI

## Code

- Install Libraries

```
pip install azure-cosmos
pip install config
pip install --upgrade openai
pip install azure-search
pip install azure-search-documents
```

- Create cog search index using cosmos db data
- Idea here is to create a cog search index using cosmos db data and then use the index to query the data using natural language
- Enable RAG pattern to search and get results
- configure open ai configuration

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxxxx"
```

- Set environment variables

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://aoainame.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "xxxxx"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"
OpenAiKey = "xxxx"
```

- Configure cognitive search

```
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

index_name = "cosmosdb-index"
# Get the service endpoint and API key from the environment
endpoint = "https://cogsearchname.search.windows.net"
key = "xxxxx"
```

- Create search client

```
credential = AzureKeyCredential(key)
client = SearchClient(endpoint=endpoint,
                      index_name=index_name,
                      credential=credential)
```

- now we need to get the search query

```
response = openai.ChatCompletion.create(
  engine="gpt4",
  messages = [{"role":"system","content":"Your job is to generate a short keyword search query based on the question or comment from the user. Only return the suggested search query with no other content or commentary. Do not answer the question or comment on it. Just generate the keywords for a search query."},{"role":"user","content":"List top candidates with environmental experience."}],
  temperature=0.0,
  max_tokens=100,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
```

- print the response

```
print(response.choices[0].message.content)
```

- assign the variable

```
searchquery = response.choices[0].message.content
```

- Query data

```
results = client.search(search_text=searchquery, select="resume/projectExperiences/details/detail,fullName", top=3)
#print(results)
candidates = []
for result in results:
    print("{}: {})".format(result["fullName"] , result["resume"]))
    candidates.append("{}: {})".format(result["fullName"] , result["resume"]))
```

- combine the rows

```
s = ''.join(str(x) for x in candidates)
```

- create prompt

```
prompttxt = "You are an AI agent, your job is to answer the question based on what content provided. If you don't know please respond don't know. Content: " + s
```

- create message

```
msg = [{"role":"system","content":  prompttxt},{"role":"user","content":"show me top candidates with environmental experience."}]
```

- now create the gpt4

```
response = openai.ChatCompletion.create(
  engine="gpt432k",
  messages = msg,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
```

```
print(response.choices[0].message.content)
```

- bring langchain

```
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
```

- save the file

```
with open('prompttxt1.txt', 'w') as f:
    f.write(prompttxt)
```

```
with open("prompttxt1.txt") as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)
```

```
from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:3]]
```

- now using gpt4 32k summarize for data

```
llm = AzureOpenAI(model_name="gpt-4-32k", deployment_id="gpt432k")
chain = load_summarize_chain(llm, chain_type="refine")
chaintext = chain.run(docs)
print(chaintext)
```

- now do the search for natural language

```
prompttxt1 = "You are an AI agent, your job is to answer the question based on what content provided. If you don't know please respond don't know. Content: " + chaintext
```

```
msg1 = [{"role":"system","content":  prompttxt1},{"role":"user","content":"show me top candidates with environmental experience."}]
```

```
response = openai.ChatCompletion.create(
  engine="gpt4",
  messages = msg1,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
```

```
print(response.choices[0].message.content)
```

- chunk the data

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

textSplitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)

docs1 = textSplitter.split_documents(docs)
```

```
print(len(docs1))
```

- now configure open ai

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxx"
```

- now natural language search

```
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import AzureOpenAI
llm = AzureChatOpenAI(model_name='gpt-35-turbo', temperature=0, deployment_name='gpt-35-turbo')
```

- now run the chain

```
from langchain.prompts import PromptTemplate
# While we are using the standard prompt by langchain, you can modify the prompt to suit your needs
promptTemplate = """You are an AI assistant tasked with summarizing documents. 
        Your summary should accurately capture the key information in the document while avoiding the omission of any domain-specific words. 
        Please generate a concise and comprehensive summary that includes details. 
        Ensure that the summary is easy to understand and provides an accurate representation. 
        Begin the summary with a brief introduction, followed by the main points. 
        Please remember to use clear language and maintain the integrity of the original information without missing any important details:
        {text}

        """
customPrompt = PromptTemplate(template=promptTemplate, input_variables=["text"])
#llm = AzureOpenAI(model_name="gpt-35-turbo", deployment_id="gpt-35-turbo")
#llm = AzureOpenAI(model_name="gpt-4", deployment_id="gpt4")

chainType = "map_reduce"
#chainType = "refine"
summaryChain = load_summarize_chain(llm, chain_type=chainType, return_intermediate_steps=True, 
                                    map_prompt=customPrompt, combine_prompt=customPrompt)
summary = summaryChain({"input_documents": docs1}, return_only_outputs=True)
outputAnswer = summary['output_text']
print(outputAnswer)
chaintext1 = outputAnswer
```

- now do the search
- Create the prompt

```
prompttxt2 = "You are an AI agent, your job is to answer the question based on what content provided. If you don't know please respond don't know. Content: " + chaintext1
msg2 = [{"role":"system","content":  prompttxt2},{"role":"user","content":"show me top candidates with environmental experience."}]
``

- now run the search

```
response = openai.ChatCompletion.create(
  engine="gpt4",
  messages = msg2,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
```

- print the output

```
print(response.choices[0].message.content)
```