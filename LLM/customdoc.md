# Using custom document with LLM (Azure Open AI/Open AI)

## Use Case

- Do we need fine tuning?
  - In most cases answer is no
  - Large Language models can be used as is
- How to consume my companies document and use it for LLM?
- How do i use my own documents with Azure Open AI

## Introduction

- In this article we will see how to use custom document with LLM (Azure Open AI/Open AI)
- 2 different process
- First is a batch process to create embeddings of the custom documents
- Second is a real time process to search the custom documents
- Second process can be either chat gpt style or GPT 3 (QnA style)
- Goal here is to clarify the process and not the code
- Not every use case needs fine tuning
  
## Architecture

- Here is the architecture
- Process defined end to end for high level understanding

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/LLM/images/LLMCustom.jpg "Architecture")

## Explanation

### Batch Process

- First get all the documents to be used for LLM
- Prefer bring a central storage like blob storage
- You can also split the sources and do each source at a time
- Documents can be in word, pdf, txt, html, etc
- Run the documents through a process to split pages, This helps in chunking the document
- Then chunk the document to 1500 characters (Only if the document is large)
- Then use Azure Open AI embedding api's to get the embeddings for the chunked documents
- Save the chunked documents and embeddings vector database or vector search
- Repeat the process for all data sources which needs to be provided for search
- Batch duration depends on how source documents are changed and make your decision how often to run the batch process

### User Interaction Process

- User interacts with the system
- They might have a nice UI or a chat bot
- User interactions are done using natural language
- First the user types some question in natural language
- First if the text is large, try to chunk it to 1500 characters
- Then send that to Azure Open AI embedding api's to get the embeddings for the chunked documents
- Next send the vector to Vector Search or DB to get the top 10 documents
- Now you can also retrieve the text associated with chunk and display
- Take top 5 documents and send it to Azure Open AI to Summarize and provide only the summary