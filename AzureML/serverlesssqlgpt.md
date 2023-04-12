# Azure open ai GPT on Azure synapse analytics serverless sql

## Write natural language queries to process data in Azure Synapse Analytics serverless SQL pool

## Pre-requisites

- Azure Account
- Azure synapse analytics
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
toolkit = SQLDatabaseToolkit(db=db)
```

- create the agent

```
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
```

- Create the agent to run queries

```
agent_executor.run("List the tables in the database")
```

- show top 10 records

```
agent_executor.run("Show top 10 records")
```

- output

```
"The top 10 records from the custviewaggrmonth table are: (11, 28, datetime.date(1996, 11, 28), '7332239.0', Decimal('1147455.40'), None, None, None, 0, None), (6, 20, datetime.date(1996, 6, 20), '7914997.0', Decimal('917464.10'), None, None, None, 0, None), (5, 15, datetime.date(1996, 5, 15), '33881812.0', Decimal('1118722.64'), None, None, None, 0, None), (2, 17, datetime.date(1996, 2, 17), '42126119.0', Decimal('145395.11'), None, None, None, 0, None), (7, 1, datetime.date(1996, 7, 1), '46844075.0', Decimal('1471253.21'), None, None, None, 0, None), (3, 12, datetime.date(1996, 3, 12), '55810084."
```