# Process Large text from pdf using Azure Open AI

## Large text processing chunks and summarize

## Pre-requisites

- Azure subscription
- Azure Machine Learning Workspace
- Document in pdf format
- Use python

## Code

- Import libraries

```
from pdfreader import SimplePDFViewer
from typing import Container
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from azure.storage.blob import ResourceTypes, AccountSasPermissions
from azure.storage.blob import generate_account_sas    
from datetime import *

today = str(datetime.now().date())
print(today)
```

- Function to read pdf and create a dataframe

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
                viewer = SimplePDFViewer(fd.read())
                all_pages = [p for p in viewer.doc.pages()]
                number_of_pages = len(all_pages)
                page_strings = ""
                print(number_of_pages)
                for page_number in range(1, number_of_pages + 1):
                    viewer.navigate(int(page_number))
                    viewer.render()
                    page_strings += " ".join(viewer.canvas.strings).replace('     ', '\n\n').strip()

                if len(page_strings) > 0:
                   df = df.append({ 'text' : page_strings}, ignore_index = True)  
            except (RuntimeError, TypeError, NameError):
                pass
```

- Show all cloumns

```
pd.set_option('display.max_colwidth', None)
```

- count the dataframe

```
df.count()
```

- Setup open ai services

```
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://oairesourcename.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxx"
```

- Load libraries

```
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

- Create a functions chunks

```
def chunks(s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]
```

- Now process and create a list

```
def buildchunks(nums, n):
    strlines = []
    i = 0
    for chunk in chunks(nums, n):
        #print(chunk , '\n')
        strlines.append(chunk)
    return strlines
```

- Function to process each lines and summary

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

- Copy the dataframe to a new dataframe

```
df1 = df.copy()
```

- Process the text and save summary in new column

```
df1['summary'] = df1["text"].apply(lambda x : processsummary(x))
```

- Display the summary

```
display(df1['summary'])
```

- Data is loaded


```
0     n  section 3 of this order, shall use the Principles to guide their development  and procurement of AI-enabled capabilities, products, and services for nonnational  security and defense purposes.  Summary: This executive order establishes principles and a process for implementing these principles to ensure the appropriate use of artificial intelligence in the federal government that is consistent with applicable law and US values and benefits the public. Agencies are encouraged to continue to use AI and the Department of Defense and Office of the Director of National Intelligence have established guidelines and principles for its use in national security and defense.....\n\nSection 8 of this order states that when considering designing, developing, acquiring and using AI in government, be guided by the common set of principles set forth in section 3 of the order which are designed to foster public trust, protect the nation's values, and ensure the use of AI remains consistent with all applicable laws. The policy of the United States is that these principles shall be governed by common policy guidance issued by OMB. Section 3 lists 9 principles for the use of AI in government including Lawful and Respectful of our Nation's Values, Purposeful and Performance Driven, Accurate, Reliable, and Effective, Safe, Secure, and Resilient, Understandable, Responsible and Traceable, Regularly Monitored, Transparent, and Accountable.ies to acquisitions of AI by agencies. As provided for by  applicable law, AI contractors may be required to meet criteria that further  the implementation of the Principles in this order. Agencies shall use  the guidance issued pursuant to section 4 of this order when establishing  contractor-facing criteria.  (d) This order does not apply to: (1) Open source software; (2) Access to  public data sets; or (3) Standards used by manufacturers.\n\nThis Executive Order requires Federal Agencies to participate in interagency bodies and promote responsible use of AI consistent with the order. The Presidential Innovation Fellows Program will attract experts from industry and academia to work within the agencies for certain periods of time, and the Office of Personnel Management will create an inventory of government rotational programs to increase the number of employees with AI expertise. Each agency is legally required to specify responsible officials to coordinate with the Agency Data Governance Body and other relevant parties. The scope of the Order applies to AI designed, developed, acquired, or used by agencies for mission execution, decision making, or providing benefit to the public, except for open source software, access to public datasets, or standards used by manufacturers.78943 Presidential Document Summarize: This executive order provides agencies with guidance for the use of Artificial Intelligence (AI) in the federal government. It outlines a set of principles and requirements that must be met when using AI, including ensuring safety and security, protecting civil rights, civil liberties, and American values, using data inputs to train AI, procuring AI applications, and adhering to relevant defense and national security guidelines. It also outlines three categories of uses where the executive order does not apply.
3Summarizing, automated systems used in development should be carefully examined for potential bias and other harms such as misuse or abuse of data. Developers should follow guidance to reduce risk and provide clear explanations of purpose of data collection and access control. Human input should be incorporated where needed and safeguards should be in place to protect individuals’ privacy, such as the ability to opt out and exercise data portability rights. A combination of transparency, accountability, and human oversight is essential to ensure ethical and fair application of such systems.
4 [UN C L ASS I F I ED   UN C L ASS I F I ED   1   U.S. DE PAR TME NT O F DEFENS E   RESPO N SI BLE  A RTI FICI AL  INT ELLI GENC E  S TRAT EGY   A ND  IMPLEM ENTA TI ON P A THWA Y   P re pa re d b y  th e  DoD R e s ponsi ble A I  W orkin g  C o unc il  in ac c ord a nc e  with  the me mora ndum  issued by  Deputy Secretary  of Defense  Kathleen  Hicks on May  26, 2021,  Implementing  Responsible Artificial  Intelligence  in the  Department of Defense.  June  2022]
5  \n\nIn summary, AI is a rapidly advancing field with implications for the national economy, security, and society. Multiple organizations are continuing to explore the potential of AI technologies, as well as the ethical and policy issues that come with them. Data privacy and security threats must be addressed in order to ensure progress, and regulations have been set up to mitigate these risks. The US is still leading in certain fields such as Machine Learning and Computer Vision, and is seeking ways to access more data so it can remain competitive in the global market. Ethical frameworks have also been developed by various government, industry, and academic entities, providing guidance for responsible AI development.
6\n\nEvaluating privacy settings and monitoring credit reports are essential steps to take in order to protect against identity theft. Keeping up to date with changes in the digital world can help prevent becoming a victim to cybercrime and knowing the risks of identity theft is an important factor in avoiding becoming a target. Taking these extra precautions can ensure that one's personal information remains secure.
Name: summary, dtype: object
```

- Save the file

```
df1.to_csv('longtextwithsummary.csv', header=True, index=False)
```