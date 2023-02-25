# Process Large text from pdf using Azure Open AI and Azure Form Recognizer

## Use Form Recognizer to extract Large text and chunk and summarize

## Pre-requisites

- Azure subscription
- Azure Machine Learning Workspace
- Document in pdf format
- Use python

## Code

- install azure ai form recognizer

```
pip install azure.ai.formrecognizer
```

- import libraries

```
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
```

- COnfigure open ai

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://aoiresourcename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "key"
```

- Summary function

```
def getsummary(mystring):
    response = openai.Completion.create(
    engine="davinci003",
    prompt= 'Summarize ' + mystring,
    temperature=0.9,
    max_tokens=300,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1
    )
    return response.choices[0].text
```

- Set form recognizer key and endpoint

```
endpoint = "https://formrecogsvc.cognitiveservices.azure.com/"
key = "key"
```

- Initialize the Form Recognizer document api client

```
document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
```

- Load all the pdf and create a dataframe

```
import os
# assign directory
directory = 'docs'
import pandas as pd

df = pd.DataFrame()
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    if (not filename.startswith(".")):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            try:
                fd = open(f, "rb")
                #viewer = SimplePDFViewer(fd.read())
                poller = document_analysis_client.begin_analyze_document("prebuilt-document", fd)
                result = poller.result()
                #print(result)
                page_strings= ""
                for page in result.pages:
                    #print(page)
                    for doc in page.lines:
                        #print('document', doc.content)
                        #page_strings += " ".join(doc.content).replace('  ', '\n\n').strip()
                        page_strings += "".join(doc.content)
                    #print(page_strings)

                if len(page_strings) > 0:
                   df = df.append({ 'text' : page_strings}, ignore_index = True)  
            except (RuntimeError, TypeError, NameError):
                pass
```

- set column to view all

```
pd.set_option('display.max_colwidth', None)
```

- Let's count the dataframe

```
df.count()
```

- Let's see the dataframe

```
df
```

- Let's chunk the text and summarize
- Build functions to chunk
  
```
def chunks(s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]
```

- Build chunks list

```
def buildchunks(nums, n):
    strlines = []
    i = 0
    for chunk in chunks(nums, n):
        #print(chunk , '\n')
        strlines.append(chunk)
    return strlines
```

- Build summary

```
def processsummary(s):
    n = 4000
    summarytext = ""
    strlines = buildchunks(s,n)
    if(len(strlines) > 1):
        summarytext = ""
        for line in strlines:
            summarytext += getsummary(line)
        summarytext1 = buildchunks(summarytext,n)
        #summarytext = buildchunks(strlines,n)
        if(len(summarytext1) > 1):
            summarytext2 = ""
            for line1 in summarytext1:
                summarytext2 += getsummary(line1)
            #finaltext = buildchunks(summarttext,n)
            finaltext = getsummary(summarytext2)
            summarytext = finaltext
        else:
            summarytext = summarytext
    else:
        summarytext = strlines
    
    return summarytext
```

- Copy to another dataframe

```
df1 = df.copy()
```

- Let's process the summary

```
df1['summary'] = df1["text"].apply(lambda x : processsummary(x))
```

- Let's see the dataframe

```
display(df1['summary'])
```