# Azure Open AI SDK Version 1 and above - Using Gpt 3.5 or 4 models

## Chatgpt example - Using GPT 3.5 or 4 models with open ai sdk version 1.0.0 and above

## Code

- install the open ai sdk version 1.0.0 and above

```
%pip install --upgrade openai
```

- Show version

```
%pip show openai
```

- Output 

```
Name: openai
Version: 1.3.5
Summary: The official Python library for the openai API
```

- Import the open ai sdk

```
import openai
```

- Set the open ai key

```
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = "https://aoaiservicename.openai.azure.com/", 
  api_key="xxxxxxxxxxxxxxxxxxxxx",  
  api_version="2023-09-01-preview"
)
```

- Set the prompt

```
message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},
{"role": "user", "content": "what is the age of michael jordan"}]
```

- Set the open ai client and parameters and call the completion

```
response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=message_text
)

print(response.choices[0].message.content)
```

- Done