# Prompt flow custom RAG QnA End to End using prmptflow CLI

## Using Azure machine learning create LLM Application and Evaluation

## Use Case

- Create a New LLM Applicatiion
- Using RAG Pattern
- Using Talk to your own data
- Using Prompt flow UI to create Data chunking and build Azure cognitive vector index.
- Using the UI to create the Standard flow
- Using one of the run evaluate the model

## Pre-requisite

- 1. Create a new LLM application using UI with your own data and will create a standard flow

![PromptflowApp](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/prompflow1.md "Architecture")

- 2. From the new Flow run evaluate the model (batch evaluate) and will create a evaluation flow
- Follow the screen with new data sample provided in this code repo

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/LLM/images/promptflow-2.jpg "Architecture")

- 3. Now Download the files for LLM Application

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/LLM/images/promptflow-1.jpg "Architecture")

- 4. Download the files for Evaluate flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/LLM/images/promptflow-3.jpg "Architecture")

## Steps to Create End to End CLI deployment

- I am using Visual Studio Code
- First create a python virtual environment with python 3.10 or more
- Set the default AML workspace
- Code is provided in this repo - 
- there should 3 folders one for standard flow, evaluate flow and then deploy.
- in the profileindex3-sample-flow there is 2 new files created
  - run.yml
  - run_evaluation.yml
- in run.yaml make sure the connection is changed to your flow connection
- Make sure data.jsonl file is available with sample data, which also has the ground truth in answers
- Make sure the embedding model and gpt models are specified based on the connection and their deploymentnames.
- In run_evaluation.yaml make sure the connection node name is changed to your evaluation flow connection
- Now in Run evaluation first you need to run the run.yaml to get the run id

```
pip install -r ./requirements.txt
```

```
pfazure run create --file run.yml --resource-group rgname --workspace-name amlworkspace
```

- Wait until it runs the flow and in terminal you should see a name like this
- Make sure Resource group name and aml workspace

```
"name": "profileindex3_sample_flow_variant_0_20231113_133240_846251",
```

- you will also see other details of the flow, which we don't need it at this time.
- Copy the name value and paste it in the run_evaluation.yaml file in the runid node
- for flow directory speficfy - ../profileindex3-sample-flow-QnA Relevance Evaluation-202311131325
- now run the evaluation

```
pfazure run create --file run_evaluation.yml --resource-group rgname --workspace-name amlworkspace
```

- Let's see the results in the terminal

```
{
    "id": "profileindex3_sample_flow_variant_0_20231113_133240_846251",
    "name": "profileindex3_sample_flow_variant_0_20231113_133240_846251",
    "status": "Succeeded",
    "created": "2021-11-13T13:32:40.000000+00:00",
    "modified": "2021-11-13T13:32:40.000000+00:00",
    "flow": {
        "id": "profileindex3_sample_flow",
        "name": "profileindex3_sample_flow",
        "version": "20231113_133240_846251"
    }
```

- if you want details
- use the below command

```
pfazure run show-metrics --name profileindex3_sample_flow_qna_relevance_evaluation_202311131325_variant_0_20231113_140049_682750 --resource-group rgname --workspace-name amlworkspace
```

## Now deployment

- Create a model file

```
az ml model create --file model.yaml --resource-group rgname --workspace-name amlworkspace
```

- Now Create endpoint

```
az ml online-endpoint create --file endpoint.yaml --resource-group rgname --workspace-name amlworkspace
```

- Now Create deployment

```
az ml online-deployment create --file deployment.yaml --all-traffic --resource-group rgname --workspace-name amlworkspace
```

- Now show the endpoint
  
```
az ml online-endpoint show -n profileweb-endpoint --resource-group rgname --workspace-name amlworkspace
```

- to get deployment logs

```
az ml online-deployment get-logs --name blue --endpoint profileweb-endpoint --resource-group rgname --workspace-name amlworkspace
```

- Now invoke the endpoint
  
```
az ml online-endpoint invoke --name profileweb-endpoint --request-file sample-request.json --resource-group rgname --workspace-name amlworkspace
``````

- Now delete the endpoint
  
```
az ml online-endpoint delete --name profileweb-endpoint --resource-group rgname --workspace-name amlworkspace
```