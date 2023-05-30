# Azure Open AI LLM with Langchain

## Introduction

- Write a code to use Azure open AI as LLM with langchain
- This code is based on [Azure Open AI LLM]
- This code is based on [Langchain](https://python.langchain.com/en/latest/modules/models/llms/integrations/azure_openai_example.html)

## Code

- First install open ai

```
pip install openai
```

- Second, import open ai and set the api type, api base, api version and api key

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoairesourcename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "key"
```

- Now set the environment variable

```
import os
os.environ["OPENAI_API_KEY"] = "key"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://aoairesourcename.openai.azure.com/"
```

- Now test with sample deployment i am using davinci is text-davinci-003

```
import openai

response = openai.Completion.create(
    engine="davinci003",
    prompt="This is a test",
    max_tokens=5
)
```

- now test AzureOpenAI as LLM

```
# Import Azure OpenAI
from langchain.llms import AzureOpenAI
```

```
# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name="davinci003",
    model_name="text-davinci-003", 
)
```

- Run the LLM

```
# Run the LLM
llm("Tell me a joke")
```

- Print the LLM

```
print(llm)
```