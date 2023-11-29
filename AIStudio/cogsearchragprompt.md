# Azure AI Studio - Talk to you data - Build your LLM Application using Azure AI Search and Prompt Flow

## Introduction

- Build your own Retrieval augeumented LLM application using Azure AI Search and Prompt Flow
- Bring your own data
- We are using pdf files for this excerise
- Vector index will be stored in Azure AI Search
- Prompt flow will be the development tool and environment used to build the application
- if the application is satisfying your needs, you can deploy it to production using managed endpoints

## Requirements

- Azure Subscription
- Azure Open AI Service
- Azure AI Studio
- Azure AI Search
- Azure Storage
- Your data in pdf format

## Steps

- First get access to https://ai.azure.com (this is our new AI Studio)
- Create a new project
- When you create a new project, you will be asked to provide Azure open ai resource information
- Provide the information where you have created the resource azure open ai resource and has TPM available to be used
- Once the project is created, you will be able to see the project dashboard
- Click on the project dashboard and you will be able to see the project information
- You should see the below Screen

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt1.jpg "Architecture")

- LLM application usualy is 2 Step process
- Step 1: Create vecotor index
- Step 2: Build the application with prompt engineering

### Step 1: Create vector index

- First click on the Indexes menu item on the left.
- Click New Index

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt2.jpg "Architecture")

- Then if you already have a index select that or click on add connection

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt3.jpg "Architecture")

- Then select the connection type
- Fill details to what blob stroage account you want to use for pdf docs
- Provide the container name
- ALso provide authentication keys
- Provide a name for the vector index connection
- then click Create Connection

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt4.jpg "Architecture")

- or you can also upload the files from local machine
- I am uploading the files from local machine, see image below

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt5.jpg "Architecture")

- click next
- Provide the AI Search or create a new one
- You can also select Azure AI Search and will be prompted to create a new one

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt6.jpg "Architecture")

- Next Select the Azure Open Connection to use
- In my case i am using Default Connection
- This is used for embeddings using ada version 2 model

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt7.jpg "Architecture")

- now give a name for the index

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt8.jpg "Architecture")

- Click Next and validate the settings and Click Create
- Now wait until Index is created
- Once it created then click on the index and see the details page

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt9.jpg "Architecture")

### Step 2: Build the application with prompt engineering

- Now the easiet way to create the application is to use the prompt flow
- Click on Example Prompt Flow and system will automatically create one for you

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt10.jpg "Architecture")

- System will take you prompt flow it created and here is a sample

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt11.jpg "Architecture")

- there few things we need to set. Look for all Azure open ai connection and select the model to use
- I am using gpt-4-turbo
- in modify_query_with_history section

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt12.jpg "Architecture")

- now check the embedding section

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt13.jpg "Architecture")

- here is the prompts i am using

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt14.jpg "Architecture")

- here is the other Azure open ai connection

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt15.jpg "Architecture")

- Now click Chat and you will be able to chat with the system
- Sample outputs

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt16.jpg "Architecture")

- Another sample

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt17.jpg "Architecture")

### Evaluate the application

- Now we are going to evaluate the prompt flow application
- Here we can change the prompts and then evaluate and validate the results
- I am using the in built evaluation
- IN your prompt flow on righ top next to chat button there is evaulate button click that.

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt18.jpg "Architecture")

- Give a name for the evaluation
- Select the prompt flow runtime to use
- For this test select question and answer pair
- In the next few sections we will see the data set needed
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt19.jpg "Architecture")

- Now select the flow application name to evaulate
- Select the node : Prompt variants
- Select the variants, this is nothing but different configutation for various models to use

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt20.jpg "Architecture")

- Now select the available metrics to calculate
- Groundness, Relevance, Coherence are the metric available
- Select the model to use, in my case i am using gpt4

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt21.jpg "Architecture")

- Next add dataset to validate against.
- i am select uploading a local dataset
- please check the schema because you need to provide new columns like question, chat_history, answer, context
- Select Add new dataset, if you already have one use the one you have
- This dataset should have ground truth as answer.

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt22.jpg "Architecture")

- Upload the files and assign the corresponding columns

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt23.jpg "Architecture")

- now check all the setting and click Submit

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt24.jpg "Architecture")

- wait for the evaluation to complete
- On the left menu go to Evaluation and you will see the evaluation you submitted

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt25.jpg "Architecture")

- Check one of the evaluation and you will see the results

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AIStudio/images/cogsearchragprompt26.jpg "Architecture")

- Done.
- Have fun creating new llm applications using prompt flow in AI Studio