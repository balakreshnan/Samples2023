# Azure open ai GPT on Azure synapse analytics serverless sql

## Write natural language queries to process data in Azure Synapse Analytics serverless SQL pool

## Pre-requisites

- Azure Account
- Azure synapse analytics
- Azure machine learning service
- Azure open ai service
- langchain 0.0.136 is the version sql works, 0.137 has breaking changes


## Azure Synapse Analytics server less sql

- Create a Database
- Load some data into underlying storage
- Create a master key
- Create identity using managed identity (this is used to access the underlying data)
- i am using tpch data as sample basically orders and lineitem

```
CREATE LOGIN sqluser WITH PASSWORD = 'password';

CREATE USER sqluser FROM LOGIN sqluser;

ALTER ROLE db_owner ADD member sqluser;


CREATE MASTER KEY ENCRYPTION by PASSWORD = 'password';
Go

CREATE DATABASE SCOPED CREDENTIAL synpasedl
WITH IDENTITY = 'Managed Identity';
GO
CREATE EXTERNAL DATA SOURCE tpch2
WITH (    LOCATION   = 'https://storagename.dfs.core.windows.net/root/tpchoutput2/',
          CREDENTIAL = synpasedl
)

CREATE EXTERNAL FILE FORMAT [SynapseParquetFormat]
       WITH ( FORMAT_TYPE = PARQUET)
GO

CREATE EXTERNAL TABLE custviewaggrmonth ( [month] INT,
    [day] INT ,
    [O_ORDERDATE] DATE,
    [C_CUSTKEY] varchar(100),
    [Tprice] DECIMAL(10,2),
    [Tdicount] DECIMAL(10,2),
    [Tqty] DECIMAL(14, 2),
    [Ttax] DECIMAL(10,2),
    [Titem] INT,
    [Textprice] DECIMAL(19, 2))
WITH ( LOCATION = '/*/*.parquet',
       DATA_SOURCE = [tpch2],
       FILE_FORMAT = [SynapseParquetFormat] )
```

- Note we are using manage identity to access since sql user won't have storage permission

## Code

- Install libraries

```
pip install pyodbc
pip install pymssql
```

- now include necessary libraries

```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.llms.openai import AzureOpenAI
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
```

```
import logging, json, os, urllib
#import azure.functions as func
import openai
from langchain.llms.openai import AzureOpenAI
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
```

- Open ai configuration

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://aoiresourcename.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxx"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"
OpenAiKey = "xxxxxxxxxxxxxxxxxx"
```

- Create llm model

```
llm = AzureOpenAI(deployment_name="deploymentname", model_name="text-davinci-003", openai_api_key=OpenAiKey)
```

- now configure database info

```
db = SQLDatabase.from_uri("mssql+pymssql://user:password@ondemandname.sql.azuresynapse.net/dbname", include_tables=['custviewaggrmonth'])
```

- now create the sql db chain

```
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

- Now run some queries

```
db_chain.run("List the tables in the database?")
```

- more queries

```
db_chain.run("show me top 10 records with total price group by month and day and show me in a table format?")
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/sqldbchain1.jpg "Architecture")

- top 10 customer key

```
db_chain.run("show me top 10 custkey records with total price group by month and day?")
```