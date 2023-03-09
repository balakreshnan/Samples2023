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

- cleanup functions

```
### Define function
def remove_special_characters(df_column,bad_characters_list):
    clean_df_column = df_column
    for bad_char in bad_characters_list:
        clean_df_column = clean_df_column.str.replace(bad_char,' ')
        print("row changes in column " + str(df_column.name) + " after removing character " + str(bad_char) + ": " ,sum(df_column!=clean_df_column))
    clean_df_column = clean_df_column.str.title()
    return clean_df_column

def remote_non_utf8(name):
     return re.sub(r'[^\x00-\x7f]',r' ',name)

### Define function
def remove_consecutive_spaces(df_column):
    clean_df_column = df_column.replace('\s\s+', ' ', regex=True)
    print("row changes in column " + str(df_column.name) +": " ,sum(df_column!=clean_df_column))
    return clean_df_column
```

- Build summary

```
def processsummary(s):
    n = 4000
    summarytext = ""
    strlines = buildchunks(s,n)
    print('length ', len(strlines))
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
        
    print('summarttext ', summarytext)
    
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

- clear non utf8 characters

```
df1['summary'] = df1['summary'].apply(remote_non_utf8)
```

- clear special characters

```
### Run function
bad_chars_lst = ["*","!","?", "(", ")", "-", "_", ",", "\n", "\\r\\n", "\r"]
df1['summary'] = remove_special_characters(df1['summary'],bad_chars_lst)
df1['text'] = remove_special_characters(df1['text'],bad_chars_lst)
display(df1[["summary"]].head(20))
```

- Remove consecutive spaces

```
df1['text'] = remove_consecutive_spaces(df1['text'])
df1['summary'] = remove_consecutive_spaces(df1['summary'])
``

- Let's see the dataframe

```
display(df1['summary'])
```