# Azure Open AI to write Job description

## Introduction

- Create a job description using Azure Open AI
- Create job description based on industry reference content
- GPT 3.5 or 4
- Using Python code
- Using Azure Machine Learning

## Code

- now let load the libraries

```python
import pandas as pd
```

- load the excel file

```
ds = pd.read_excel('data/sample.xlsx', index_col=0, header=0) 
```

- get unique profile

```
ds['New Job Profile '].unique()
```

- Load environment variables

```
from dotenv import dotenv_values
# specify the name of the .env file name 
env_name = "env2.env" # change to use your own .env file
config = dotenv_values(env_name)
```

- import open ai config

```
import openai 

openai.api_type = "azure"
openai.api_key = config["AZURE_OPENAI_API_KEY"]
openai.api_base = config["AZURE_OPENAI_ENDPOINT"]
openai.api_version = config["AZURE_OPENAI_API_VERSION"]
```

- create split text function 

```
def split_text(text):
    max_chunk_size = 2048
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
```

- Create summary function for gpt 3.5 16K

```
def generate_summary1(text, reference):
    input_chunks = split_text(text)
    output_chunks = []
    msg = f"Summarize the following responsibilites into 5 bullet point:\n{text}, Use reference document to summarize: {reference} Be concise, natural, balanced and consistence response.\n\n \n\nSummary:"
    #msg = f"Please summarize 5 different separate summarization on the following text:\n{chunk}\n\n Be concise and provide natural and balanced response.\n\nSummary:"
    message = [{ "role" : "system", "content" : "You are an AI assistant that helps people find information."},
    { "role" : "user", "content" : msg}]
    response = openai.ChatCompletion.create(
        model="gpt-35-turbo-16k",
        deployment_id="gpt-35-turbo-16k",
        #prompt=(f"Please summarize the following text:\n{text}\n\nSummary:"),
        messages=message,
        temperature=0.5,
        max_tokens=1024
    )
    #print(response)
    #summary = response.choices[0].message.strip()
    summary = response.choices[0].message.content

    return "".join(summary)
```

- process dataframe rows

```
i = 0
for row in ds['New Job Profile '].unique():
    #print(row)
    print(i)
    i = i + 1
    df1 = pd.DataFrame(ds.loc[ds['New Job Profile '] == row])
    #print(type(df1))
    allsumtext = ""
    indstr1 = ""
    for index, row1 in df1.iterrows():
        #print(type(row1))
        #print(row1[2])
        sumtext = ""
        indstr = row1[6]
        indstr1 = indstr
        
        #print(df1.count())
        sumtext = str(row1[5])
        allsumtext = allsumtext + sumtext
        #print(allsumtext)
    stext = generate_summary1(allsumtext, indstr1)
    #print(' Profile --->', indstr1)
    print(' profile: ', row)
    print(stext, '\n')
    #print(' Profile --->', indstr1)
    #print(' profile: ', row)
    with open("myfile.csv", 'a') as file1:
        ftxt = indstr1 + "," + stext
        file1.write(ftxt)
        file1.write('\n')
```

- Now to process in GPT 4

```
import openai 

openai.api_type = "azure"
openai.api_key = "xxxxx"
openai.api_base = "https://aoainame.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
```

- function

```
def generate_summary4(text, reference):
    input_chunks = split_text(text)
    output_chunks = []
    #msg = f"Summarize the following responsibilites into 5 bullet point:\n{text}, Use reference document to summarize: {reference} Be concise, natural, balanced and consistence response and number the bullet point.\n\n \n\nSummarize as JSON Output:"
    #msg = f"Please summarize 5 different separate summarization on the following text:\n{chunk}\n\n Be concise and provide natural and balanced response.\n\nSummary:"
    msg = f"""Summarize the following responsibilites into 5 bullet point:\n{text}, 
    Use reference document to summarize: {reference} 
    Be concise, natural, balanced and consistence response and number the bullet point.
    Summarize as JSON Output:"""
    
    message = [{ "role" : "system", "content" : "You are an Job assistant that helps people write job descriptions."},
    { "role" : "user", "content" : msg}]
    response = openai.ChatCompletion.create(
        model="gpt-4-32k",
        deployment_id="gpt-4-32k",
        #prompt=(f"Please summarize the following text:\n{text}\n\nSummary:"),
        messages=message,
        temperature=0.5,
        max_tokens=1024
    )
    #print(response)
    #summary = response.choices[0].message.strip()
    summary = response.choices[0].message.content

    return "".join(summary)
```

- process dataframe rows

```
i = 0
for row in ds['New Job Profile '].unique():
    #print(row)
    print(i)
    i = i + 1
    df1 = pd.DataFrame(ds.loc[ds['New Job Profile '] == row])
    #print(type(df1))
    allsumtext = ""
    indstr1 = ""
    for index, row1 in df1.iterrows():
        #print(type(row1))
        #print(row1[2])
        sumtext = ""
        indstr = row1[6]
        indstr1 = indstr
        
        #print(df1.count())
        sumtext = str(row1[5])
        allsumtext = allsumtext + sumtext
        #print(allsumtext)
    stext = generate_summary4(allsumtext, indstr1)
    #print(' Profile --->', indstr1)
    print(' profile: ', row)
    print(stext, '\n')
    #print(' Profile --->', indstr1)
    #print(' profile: ', row)
    with open("myfile.csv", 'a') as file1:
        ftxt = indstr1 + "," + stext
        file1.write(ftxt)
        file1.write('\n')
```