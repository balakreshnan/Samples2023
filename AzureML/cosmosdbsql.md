# Azure Cosmos DB DBChain concept

## Create SQL from Natural language

## Pre-requisites

- Azure Account
- Azure Cosmos db
- Azure machine learning service
- Azure Open AI

## Code

- Install Libraries

```
pip install azure-cosmos
pip install config
pip install --upgrade openai
```

- import libraries

```
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
from azure.cosmos import ThroughputProperties

import config
```

- Configure cosmos db

```
HOST = "https://cosmosdbname.documents.azure.com:443/"
MASTER_KEY = "xxxxxxxxxxxxxxxxxxxxxxx=="
DATABASE_ID = "dbnamepeople"
CONTAINER_ID = "people1"
```

- Create cosmos db client

```
from azure.cosmos import CosmosClient

import os
URL = HOST
KEY = MASTER_KEY
client = CosmosClient(URL, credential=KEY)
```

- query data

```
DATABASE_NAME = 'dbnamepeople'
database = client.get_database_client(DATABASE_NAME)
CONTAINER_NAME = 'people1'
container = database.get_container_client(CONTAINER_NAME)

# Enumerate the returned items
import json
for item in container.query_items(
        query='SELECT * FROM mycontainer r WHERE r.id="2"',
        enable_cross_partition_query=True):
    print(json.dumps(item, indent=True))
```

- Azure openai configuration

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://aoainame.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "xxxxx"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"
OpenAiKey = "xxxxx"
```

- import libraries

```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.llms.openai import AzureOpenAI

import openai
```

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "xxxxx"
```

- now create prompt engineering to create sql

```
response = openai.ChatCompletion.create(
  engine="gpt4",
  messages = [{"role":"system","content":"You are an AI azure cosmosdb agent, your job is to create query based on questions asked with the schema provided, schema name: people1 { id: xxxx, employeeId: 11111111, employeeNumber: 11111, status: Active, firstName: xxx, middleName: x, lastName: xxxxx, initials: xxx, preferredName: null, fullName: xxxxx., displayName:xxxxx, email: xxx@xxx.com, username: xxxx, gender: Male,  maritalStatus: Married, originalHireDate: 2023-03-01T00:00:00.000Z, rehireDate: 2023-03-01T00:00:00.000Z, acquisitionDate: null, anniversaryDate: 2021-03-01T00:00:00.000Z, terminationDate: null, birthDate: 1988-04-18T00:00:00.000Z, mobile: null, directOfficePhone: +1 xxx-xxx-xxxx, yearsWithcompany: 2, yearsWithOtherFirms: 13, languages: [], certifications: [], educations: [], registrations: [], homeInfo: { address1: null, address2: null, city: null, state: null, zip: null, countryCode: null, countryName: null, phone: null, mobile: null, email: null},resume: { awards: [], miscellaneousInfos: [], professionalAffiliations: [], projectExperiences: [], publications: [], seminars: [], summaries: [], titles: []}, projects: [] }. Only respond with detail query as output"},{"role":"user","content":"show me records who have Boeing Realty experience."}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
  ```

- output

```
print(response.choices[0].message.content)
```

- run the sql now

```
import json
for item in container.query_items(
        query=response.choices[0].message.content,
        enable_cross_partition_query=True):
    print(json.dumps(item, indent=True))
```