# Azure Open AI Summarize large 70+ pdf

## Overview

- Summarize large pdf
- Load pdf
- Clean the data
- Load the data using Langchain
- Using Azure Machine learning
- Upload meeting transcript into a folder
- sample pdf is available in docs folder

## Code

- install libraries

```
pip install pdfreader
pip install langchain
pip install unstructured
pip install tiktoken
pip install faiss-cpu
pip install pypdf
```

- load libraries

```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
```

- Load environment variable for open ai configuration and keys

```
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://aoiresourcename.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxx"
os.environ["DEFAULT_EMBED_BATCH_SIZE"] = "1"
```


- Setup embeddings

```
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

llm = OpenAI(engine="davinci",temperature=0)

text_splitter = CharacterTextSplitter()
```

- now load the pdf

```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/Blueprint-for-an-AI-Bill-of-Rights.pdf")
pages = loader.load_and_split()
```

- Let's search for some information

```
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(chunk_size=1))
docs = faiss_index.similarity_search("How to create data privacy?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content)
```

- Let now configure Azure open ai embeddings

```
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#embeddings = OpenAIEmbeddings(chunk_size=1)
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="gpt-35-turbo")
```

- load the summarizer from langchain

```
from langchain.chains.summarize import load_summarize_chain
```

- Setup LLM now

```
from langchain.chat_models import ChatOpenAI
llm=ChatOpenAI(temperature=0.7, engine="chatgpt", max_tokens=300)
```

- Normal summarize the entire document

```
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

- output

```
'The White House has released a Blueprint for an AI Bill of Rights which outlines five principles to protect civil rights, civil liberties, and privacy in the age of artificial intelligence. The principles include safe and effective systems, protections against algorithmic discrimination, data privacy, human alternatives and considerations, and examples of automated systems. The Blueprint is non-binding but provides a framework for policies and practices that protect civil rights and promote democratic values in the building, deployment, and governance of automated systems. The article covers various issues related to technology and data ethics, including algorithmic bias, surveillance and privacy, and artificial intelligence.'
```

- Now prompt engineer to pull insights

```
prompt_template = """Extract action items:


{text}


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(OpenAI(engine="davinci",temperature=0), chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
chain({"input_documents": docs}, return_only_outputs=True)
```

- output of action items

```
Action Items:\n
1. Establish a federal agency to oversee the development and implementation of an AI Bill of Rights.\n
2. Create a public-private partnership to develop standards and best practices for AI systems.\n
3. Establish a national AI research and development program.\n
4. Create a national AI education and training program.\n
5. Establish a national AI ethics board.\n
6. Develop a framework for AI governance.\n
7. Establish a national AI data privacy and security framework.\n
8. Create a national AI safety and security framework.\n
9. Establish a national AI liability framework.\n
10. Develop a national AI transparency framework.\n
11. Continue to engage the public in the development of policies and practices that protect civil rights and promote democratic values in the building, deployment, and governance of automated systems.\n
12. Develop sector-specific guidance to guide the use of automated systems in certain settings.\n
13. Develop alternative, compatible safeguards through existing policies that govern automated systems and AI.\n
14. Develop implementation policies to national security and defense activities informed by the Blueprint for an AI Bill of Rights where feasible.\n
15. Develop automated systems that protect the American public in the age of artificial intelligence.
```

- Now lets ask for Recommendation to follow for building AI based applcations

```
prompt_template = """Extract Recommendations for developing ai based applications:


{text}


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(OpenAI(engine="davinci",temperature=0), chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
chain({"input_documents": docs}, return_only_outputs=True)
```

- here is the output

```
When developing AI-based applications, organizations should adhere to principles such as being lawful and respectful of values, purposeful and performance-driven, accurate, reliable, and effective, safe, secure, and resilient, understandable, responsible and traceable, regularly monitored, transparent, and accountable. Additionally, organizations should consider risk management frameworks, stakeholder engagement, and proactive measures to protect individuals and communities from algorithmic discrimination. Data privacy should be protected by design and by default, with data collection and use-case scope limits established and data retention timelines documented and justified. Entities should provide reporting on the data collected and stored about users, and should provide clear and understandable notice to the public that an automated system is being used. Human alternatives, consideration, and fallback should be provided, and training and assessment should be implemented to ensure the system is used appropriately and to mitigate the effects of automation bias.
```

- Now lets ask for what to avoid n building ai applications

```
prompt_template = """What should i avoid for building ai applications:


{text}


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(OpenAI(engine="davinci",temperature=0), chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
chain({"input_documents": docs}, return_only_outputs=True)
```

- output

```
When building AI applications, organizations should adhere to ethical principles and reporting expectations, provide clear and valid explanations, provide meaningful access for oversight, and consider the potential impacts of AI on stakeholders. Additionally, organizations should avoid practices that violate civil rights, civil liberties, and privacy, such as algorithmic discrimination, collecting sensitive data, using surveillance technology, or targeting underserved communities. Pre-deployment testing, risk identification and mitigation, and ongoing monitoring should be performed to ensure safety and effectiveness, and independent evaluation and plain language reporting should be performed and made public whenever possible.
```

- Now you can see how prompt engineering can get different prompt gives us different results.