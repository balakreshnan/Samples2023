# Azure Machine learning Fine tuning Large language models Q&A and Summarization

## Using GPU compute to fine tune large language models

## introduction

- Fine tune large language models in Azure ML
- Using GPU Clusters
- Using existing samples from our Azure machine learning sdk examples
- Using SKU - STANDARD_NC64AS_T4_V3
- I had to request quota increase using Azure ML to achieve this experiment
- using open source data set
- Here is the GPU details from url: https://www.nvidia.com/en-us/data-center/tesla-t4/
- Summarization url : https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/summarization/news-summary.ipynb
- QnA url: https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/question-answering/extractive-qa.ipynb
- I am using python 3 kernel

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune1.jpg "Architecture")

## Q&A

- Clone the sample section or download the sample from the url: https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/question-answering/extractive-qa.ipynb
- First install necesary packages
- Then run each cell in the notebook
- Check the sample data set provided in the notebook

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune2.jpg "Architecture")

- Submit the job and will take about 4 to 5 hours to complete

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune4.jpg "Architecture")

- Metrics

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune6.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune7.jpg "Architecture")

## Summarization

- Clone the sample section or download the sample from the url:https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/summarization/news-summary.ipynb
- QnA 
- First install necesary packages
- Then run each cell in the notebook
- Check the sample data set provided in the notebook

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune3.jpg "Architecture")

- Submit the job and will take about 10 to 11 hours to complete

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune5.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune8.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetune9.jpg "Architecture")