# Azure Cognitive Search with Natural Language query

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
pip install --upgrade tiktoken
```

- Create cog search index using cosmos db data
- Idea here is to create a cog search index using cosmos db data and then use the index to query the data using natural language
- Enable RAG pattern to search and get results
- configure open ai configuration

- Now import the libraries

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

- Set environment variables

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://aoainame.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "xxxxxx"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"
OpenAiKey = "xxxxx"
```

- Test the open ai working, using GPT 4

```
#test open ai 
response = openai.ChatCompletion.create(
  engine="gpt4",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"when was the model trained"},{"role":"assistant","content":"I am an AI language model, and I am constantly being updated and improved. However, my knowledge is most accurate up to September 2021, so any information or events beyond that date might not be reflected in my responses."}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message)
```

- now let's search inform

```
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

index_name = "cosmosdb-index"
# Get the service endpoint and API key from the environment
endpoint = "https://cogsearchname.search.windows.net"
key = "xxxxxxxxxxxxxxxxxxxx"
```

- setup search client

```
credential = AzureKeyCredential(key)
client = SearchClient(endpoint=endpoint,
                      index_name=index_name,
                      credential=credential)
```

- Create a search query to query from

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
print(response.choices[0].message.content)
```

- now Search the cognitive services and get results
- This is RAG retrieval augumented generation pattern
- For Cognitive search to use semantic search, we need to flatten the data and create a cog search index
- Here i am not flattening the data, but using the existing data from cosmos db


```
searchquery = response.choices[0].message.content
results = client.search(search_text=searchquery, select="resume/projectExperiences/details/detail,fullName", top=3)
#print(results)
candidates = []
for result in results:
    #print("{}: {})".format(result["fullName"] , result["resume"]))
    #candidates.append("name: {} , Experiences: {})".format(result["fullName"] , result["resume"]))
    candjson = {
        "name" : result["fullName"],
        "Experiences" : result["resume"]
    }
    candidates.append(candjson)
```

- combine the data

```
s = ''.join(str(x) for x in candidates)
```

- now generate the response

```
prompttxt = "You are an AI agent, your job is to answer the question based on what content provided. If you don't know please respond don't know. Content: " + s
```

- Create the message to chatgpt model

```
msg = [{"role":"system","content":  prompttxt},{"role":"user","content":"show me top candidates with environmental experience."}]
```

- now generate the response
- i am using gpt 4 32K to test given large data

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
print(response.choices[0].message.content)
```

- output

```
Based on the content provided, here are the top candidates with environmental experience:

1. Golightly, William D. (Bill)
   - Extensive experience in feasibility studies, remedial investigations, and remedial actions for various projects including aerospace facilities, military bases, and commercial properties.
   - Expertise in soil and groundwater remediation, vapor intrusion mitigation, and hazardous waste management.
   - Experience in regulatory negotiations and providing consulting services for various clients.

2. Almestad, Charles H. (Charlie)
   - Experience in remedial investigations, evaluation of remedial alternatives, remedial design and implementation of remedial actions.
   - Expertise in groundwater modeling, water supply studies, and soil assessments.
   - Experience in regulatory liaison, risk assessment, and providing expert witness services for litigation support.
```

- Calculate token

```
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
```

- Lets process experience and get the details only and use it for search

```

totaltokens = 0
print(len(candidates))
peoplelist = []
for cand in candidates:
    strexp = ""
    strname = ""
    for key in cand:
        
        if key == 'name':
            strname = cand[key]
        if key == 'Experiences':
            a_dict = cand[key]
            for ky in a_dict:
                for i in a_dict[ky]:
                    for j in i:
                        #print(type(i[j]))
                        tmpstr = i[j]
                        totaltokens += num_tokens_from_string(tmpstr[0]["detail"], "cl100k_base")
                        strexp +=  tmpstr[0]["detail"] + " "   
    peoplelist.append("{}: {})".format(strname , strexp))

print(totaltokens)
print(peoplelist)
```

- now generate the response with gpt 4 or gpt 3.5
- below is gpt 4

```
s = ''.join(str(x) for x in peoplelist)
```

- create the prompt

```
prompttxt = "You are an AI agent, your job is to answer the question based on what content provided. If you don't know please respond don't know. Content: " + s
```

- create a message

```
msg = [{"role":"system","content":  prompttxt},{"role":"user","content":"show me top candidates with environmental experience."}]
```

- generate the response

```
response = openai.ChatCompletion.create(
  engine="gpt4",
  messages = msg,
  temperature=0.2,
  max_tokens=500,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)
```

- output

```
Based on the content provided, the top candidates with environmental experience are:

1. William D. (Bill) Golightly: He has extensive experience in feasibility evaluations, remedial actions, environmental site assessments, and remediation of soil and groundwater impacted by various contaminants. He has worked on projects such as the Miller Children's Hospital in Long Beach, CA, and the Terminal 1 facility at San Diego Airport.

2. Charles H. (Charlie) Almestad: With a background in environmental cleanup projects, remedial investigations, risk assessments, and hazardous waste management, Charlie has managed projects involving various contaminants in soil and groundwater. He has provided environmental consulting services for numerous sites and has experience in groundwater supply studies and dewatering plans.
```

- now lets try with gpt 3.5
- create the prompt

```
prompttxt1 = "You are an AI Assistant, your job is to answer the question based on what content provided. be polite and be concise as possible. provide details to your answer in bullet point. Use the content provided: " + s
prompttxt1 += " If you don't know don't make content please respond don't know."
```

- Create a message

```
msg1 = [{"role":"system","content":  prompttxt1},{"role":"user","content":"show me top candidates with environmental experience."}]
```

- gpt 3.5 respone

``` 
response = openai.ChatCompletion.create(
  engine="gpt-35-turbo",
  messages = msg1,
  temperature=0.2,
  max_tokens=500,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
print(response.choices[0].message.content)
```

- output

```
Based on the provided content, here are some top candidates with environmental experience:

1. Almestad, Charles H. (Charlie): Project manager since 1986 for an environmental cleanup project involving the remedial investigation, evaluation of remedial alternatives, remedial design and implementation of remedial actions relating to six classes of chemicals in soil and/or groundwater in five separate areas of a 17-acre site. Also, provided environmental consulting services to Equity Office Properties (and Spieker Properties) related to hazardous materials in the soil, groundwater or structures at over 50 sites (over 70 projects) in the Northern and Southern California, Oregon, Colorado and Washington.

2. Golightly, William D. (Bill): Feasibility Study Principal engineer responsible for performing feasibility evaluations to facilitate the selection of remedial actions at a former aircraft manufacturing facility in southern California. Oversaw a remedial action work plan implementation during the construction of the new Miller Children's Hospital in Long Beach, CA. Also, served as the Principal-in Charge for Kleinfelder’s $3.5M on-call contract with the San Diego County Regional Airport Authority (SDCRAA) Facilities Development Department.

These candidates have extensive experience in environmental cleanup projects, remedial investigations, feasibility studies, and providing environmental consulting services. They have also worked on projects related to hazardous waste management, groundwater supply and quality assessment, and compliance monitoring.
```