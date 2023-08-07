# Building Large Language Models Applications using Prompt Flow in Azure ML - Simple LLMOps

## Introduction

- Build Large Language Models using Prompt Flow in Azure ML
- Using Prompt Flow
- Using Azure Cognitive search vector services
- using Azure open ai service

## Concept

- First create a vector store and provide cognitive search service
- Create a connections to open ai and also cognitive search
- Create a prompt flow
- Create a pipeline to run the prompt flow
- i am using existing vector store and cognitive search service

## Pre-requisites

- Azure Cognitive search
- Azure machine learning service
- Azure Open AI service

## Code

- Go to Azure Machine Learning services
- Create a connection for open ai in prompt flow sections

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog4.jpg "Architecture")

- Now create vector index search connections
- i am using existing azure cognitive search service
- Start with uploading data to vector store

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog1.jpg "Architecture")

- Give a name
- Upload the pdf files
- Select the cognitive search service for vector index
- Select the vector index
- Now Click Next
- In the next screen select Azure Open ai services

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog2.jpg "Architecture")

- Select compute as serverless

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog21.jpg "Architecture")

- Review and create the vector store connection

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog3.jpg "Architecture")

## Prompt flow or LLM application

- Now create a prompt flow
- Create a Standard flow

- Now create a prompt flow
- Create a Standard flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog5.jpg "Architecture")

- Some inputs will be automatically created, but you have to fill the default question
- But for evaulation we need to add more inputs like the image below

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog16.jpg "Architecture")

- Select the Azure Open AI connection
- Now add flow to search the index

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog7.jpg "Architecture")

- now generate prompt context

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog8.jpg "Architecture")

- Code

```
from typing import List
from promptflow import tool
from embeddingstore.core.contracts import SearchResultEntity

@tool
def generate_prompt_context(search_result: List[dict]) -> str:
    def format_doc(doc: dict):
        return f"Content: {doc['Content']}\nSource: {doc['Source']}"
    
    SOURCE_KEY = "source"
    URL_KEY = "url"
    
    retrieved_docs = []
    for item in search_result:

        entity = SearchResultEntity.from_dict(item)
        content  = entity.text or ""
        
        source = ""
        if entity.metadata is not None:
            if SOURCE_KEY in entity.metadata:
                if URL_KEY in entity.metadata[SOURCE_KEY]:
                    source = entity.metadata[SOURCE_KEY][URL_KEY] or ""
        
        retrieved_docs.append({
            "Content": content,
            "Source": source
        })
    doc_string = "\n\n".join([format_doc(doc) for doc in retrieved_docs])
    return doc_string
```

- now specfiy the input for the above code
- Input is shown in below image

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog9.jpg "Architecture")

- now create a flow for prompt variant
- Specify the prompt to use in the application
- Next setup to answer the question

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog10.jpg "Architecture")

- Now we add evaulation step on the LLM
- Bring python script flow into the graph
- name it line_process
- setup the inputs as below image
- use the code below
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog17.jpg "Architecture")

- python code

```
from promptflow import tool


@tool
def line_process(groundtruth: str, prediction: str):
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param prediction: the prediction of a single line.
    """

    processed_result = ""

    # Add your line processing logic here

    return processed_result
```

- Now add aggregate flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog18.jpg "Architecture")

- Setup the azure open ai connection
- select the model deployment to use, i am using chat for gpt 3.5 turbo
- Set the temperature and max token output
- Send the output as prompt_text
- you can change the prompt and run again
- Try with different variations one shot, few shot examples in prompt
- Try to chain the prompt and experiment
- Now on the right top corner click on the run button
- Wait for the run to complete
- You will see the graph in the right side

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog19.jpg "Architecture")

## Deploy

- Next Deploy the code is all is well
- Give name to the endpoint

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog12.jpg "Architecture")

- now choose the outputs for the endpoint

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog13.jpg "Architecture")

- Now select the connection and models to use
- We need embedding and also gpt 3.5 turbo model

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog14.jpg "Architecture")

- Select compute
- Provide number of instances and also the size of the instance

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/promptflowcog15.jpg "Architecture")

- Now review and then deploy
- Deployment will take few minutes to deploy the endpoint and deploy the traffic
- Test the model to make sure it is working as expected