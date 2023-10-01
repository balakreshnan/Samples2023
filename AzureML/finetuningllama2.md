# Azure Machine learning Fine tuning llama2 - Q&A and Summarization

## Using GPU compute to fine tune llama2 and using deepspeed to parallelize the fine tuning

## introduction

- Fine tune large language - Llama2 models in Azure ML
- Using GPU Clusters
- Using existing samples from our Azure machine learning sdk examples
- I had to request quota increase using Azure ML to achieve this experiment
- using open source data set
- Summarization url : https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-generation/summarization_with_text_gen.ipynb
- Text classification url: https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-classification/emotion-detection-llama.ipynb
- I am using python 3 kernel

## LLama2 - 7 billion

- Running sample for 7 billion model
- using sample data set
- using 2 A100 4 GPU computer
- SKU used - STANDARD_NC96ADS_A100_V4
- Using batch size of 1
- More than that will fail with out of memory error
- 4 bit Quantization is enabled

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2-1.jpg "Architecture")

- GPU Metrics

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2-2.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2-3.jpg "Architecture")

## LLama2 - 13 billion

- Running sample for 7 billion model
- using sample data set
- using 2 A100 4 GPU computer
- SKU used - STANDARD_NC96ADS_A100_V4
- Using batch size of 1
- More than that will fail with out of memory error
- 4 bit Quantization is enabled

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2-4.jpg "Architecture")

- GPU Metrics

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2-5.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2-6.jpg "Architecture")

## LLama2 - 70 billion