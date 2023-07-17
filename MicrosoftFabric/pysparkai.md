# Microsoft Fabric using natural language processing (NLP) to manupulate data

## Introduction

- This is a sample project to demonstrate how to use NLP to manipulate spark dataframe
- using pyspark-ai
- using langchain and open ai

## Prerequisites

- Microsoft Fabric
- MS Teams Data Set for messages
- File in JSON format

## Code

- First install open ai, langchain and pyspark-ai

```
pip install openai
pip install langchain
pip install pyspark-ai
```

- Set performance configuration

```
%%pyspark
spark.conf.set('spark.sql.parquet.vorder.enabled', 'true')
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")
```

- import libraries

```
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
```

- configure Azure Chat Open AI model

```
BASE_URL = "https://aoainame.openai.azure.com/"
API_KEY = "xxxxx"
DEPLOYMENT_NAME = "chatgpt"
AzureChatOpenAI_model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)
```

- Now setup pyspark ai model

```
from pyspark_ai import SparkAI

spark_ai = SparkAI(llm=AzureChatOpenAI_model, verbose="true")
spark_ai.activate()
```

- Now load the data set

```
messagesDF = spark.read.json("Files/Messages_Large.json")
```

- display the data set

```
display(messagesDF)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/images/pysparkai1.jpg "Architecture")

- Now lets only get the necessary columns
- Flattern the internetMessageHheaders
- Create new column for year, month and day

```
ai_df = messagesDF.ai.transform(" include bodyPreview,  createdDateTime, from, importance, internetMessageHeaders, sender, subject, toRecipients").ai.transform("add column for year, month and day using createdDateTime").ai.transform(" flatten internetMessagesHeaders and add rows with new column names")
```

- output

```
INFO: SQL query for the transform:
SELECT bodyPreview, createdDateTime, from, importance, internetMessageHeaders, sender, subject, toRecipients 
FROM temp_view_for_transform

SQL query for the transform:
SELECT bodyPreview, createdDateTime, from, importance, internetMessageHeaders, sender, subject, toRecipients 
FROM temp_view_for_transform

INFO: SQL query for the transform:
SELECT *, 
  YEAR(createdDateTime) AS year, 
  MONTH(createdDateTime) AS month, 
  DAY(createdDateTime) AS day 
FROM temp_view_for_transform

SQL query for the transform:
SELECT *, 
  YEAR(createdDateTime) AS year, 
  MONTH(createdDateTime) AS month, 
  DAY(createdDateTime) AS day 
FROM temp_view_for_transform

INFO: SQL query for the transform:
SELECT 
  bodyPreview,
  createdDateTime,
  from.emailAddress.address AS from_email_address,
  from.emailAddress.name AS from_email_name,
  importance,
  internetMessageHeaders[0].value AS header_value_1,
  internetMessageHeaders[0].name AS header_name_1,
  internetMessageHeaders[1].value AS header_value_2,
  internetMessageHeaders[1].name AS header_name_2,
  internetMessageHeaders[2].value AS header_value_3,
  internetMessageHeaders[2].name AS header_name_3,
  sender.emailAddress.address AS sender_email_address,
  sender.emailAddress.name AS sender_email_name,
  subject,
  toRecipients[0].emailAddress.address AS to_email_address_1,
  toRecipients[0].emailAddress.name AS to_email_name_1,
  toRecipients[1].emailAddress.address AS to_email_address_2,
  toRecipients[1].emailAddress.name AS to_email_name_2,
  toRecipients[2].emailAddress.address AS to_email_address_3,
  toRecipients[2].emailAddress.name AS to_email_name_3,
  year,
  month,
  day
FROM temp_view_for_transform

SQL query for the transform:
SELECT 
  bodyPreview,
  createdDateTime,
  from.emailAddress.address AS from_email_address,
  from.emailAddress.name AS from_email_name,
  importance,
  internetMessageHeaders[0].value AS header_value_1,
  internetMessageHeaders[0].name AS header_name_1,
  internetMessageHeaders[1].value AS header_value_2,
  internetMessageHeaders[1].name AS header_name_2,
  internetMessageHeaders[2].value AS header_value_3,
  internetMessageHeaders[2].name AS header_name_3,
  sender.emailAddress.address AS sender_email_address,
  sender.emailAddress.name AS sender_email_name,
  subject,
  toRecipients[0].emailAddress.address AS to_email_address_1,
  toRecipients[0].emailAddress.name AS to_email_name_1,
  toRecipients[1].emailAddress.address AS to_email_address_2,
  toRecipients[1].emailAddress.name AS to_email_name_2,
  toRecipients[2].emailAddress.address AS to_email_address_3,
  toRecipients[2].emailAddress.name AS to_email_name_3,
  year,
  month,
  day
FROM temp_view_for_transform
```

- display the output

```
display(ai_df)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/images/pysparkai2.jpg "Architecture")