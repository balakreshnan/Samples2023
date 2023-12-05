# GPT 4 turbo with your own PDF documents without chunking

## Introduction

- Sample to load pdf into gpt 4 turbo model
- There is no chunking of the document

## Requirements

- Azure Subscription
- Azure machine learning workspace
- Azure open ai resource
- Azure storage account
- Sample PDF file i am using gpt4 vision paper - https://arxiv.org/pdf/2309.17421.pdf

## Steps

### install libraries

```
%pip install PyPDF2
```

### Setup open sdk v1

```
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "xxxxxxxxxxxxxxxxxxx"
```

- invoke the open ai client

```
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = "https://aoainame.openai.azure.com/", 
  api_key="xxxxxxxxxxxxxxxxxxxxxxxx",  
  api_version="2023-09-01-preview"
)
```

- Setup the pdf information

```
import PyPDF2
import openai
import os

# Replace with your OpenAI API key and model
my_ai_model = "gpt-4-turbo"

pdf_file = "gptv4vision2309.17421.pdf"
```

- Read the pdf file

```
processed_text_list = []
# Open the PDF file in binary mode
with open(pdf_file, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    #print(pdf_reader)
    # Iterate through each page and extract text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        processed_text_list.append(page_text)

# Combine all AI-processed text into a single string
combined_text = "\n".join(processed_text_list)
```

- now format the message to send to model

```
    messages = [
        {
            "role": "system",
            "content": """You are a Assistant, a backend processor.
- User input is messy raw text extracted from a PDF page by PyPDF2.
- Answer with polite and positive sense.
"""
        },
        {
            "role": "user",
            "content": "Summarize the content:" + combined_text
        }
    ]
```

- invoke the model

```
response = client.chat.completions.create(
    model="gpt-4-turbo", # model = "deployment_name".
    messages=messages
)

print(response.choices[0].message.content)
```

- print the token usage

```
print(response.usage)
```