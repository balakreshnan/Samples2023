# Azure Machine learning Fine tuning Parallel Large language models Q&A and Summarization

## Using GPU compute to fine tune large language models and using deepspeed to parallelize the fine tuning

## introduction

- Fine tune large language models in Azure ML
- Using GPU Clusters
- Using existing samples from our Azure machine learning sdk examples
- I had to request quota increase using Azure ML to achieve this experiment
- using open source data set
- Summarization url : https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/summarization/news-summary.ipynb
- QnA url: https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/question-answering/extractive-qa.ipynb
- I am using python 3 kernel



## Runs

- using 2 A100 4 GPU computer
- Using SKU - Standard_NC96ads_A100_v4 (96 cores, 880 GB RAM, 256 GB disk)
- we are using 2 GPU VM to horizontally run the fine tuning
- Few changes to do in step 5
- Summarization url : https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/summarization/news-summary.ipynb
- in step 3 i need to add the above SKU

```
import ast

if "computes_allow_list" in foundation_model.tags:
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["computes_allow_list"]
    )  # convert string to python list
    #computes_allow_list.append("STANDARD_NC24ADS_A100_V4")
    computes_allow_list.append("Standard_NC48ads_A100_v4")
    computes_allow_list.append("Standard_NC96ads_A100_v4") 
    print(f"Please create a compute from the above list - {computes_allow_list}")
else:
    computes_allow_list = None
    print("Computes allow list is not part of model tags")
```

- to optimize the speed of the run we need change batch size in step 5

```
# Training parameters
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    metric_for_best_model="rouge1"
    #auto_find_batch_size=True
)
print(f"The following training parameters are enabled - {training_parameters}")

# Optimization parameters - As these parameters are packaged with the model itself, lets retrieve those parameters
if "model_specific_defaults" in foundation_model.tags:
    optimization_parameters = ast.literal_eval(
        foundation_model.tags["model_specific_defaults"]
    )  # convert string to python dict
else:
    optimization_parameters = dict(
        apply_lora="true", apply_deepspeed="true", apply_ort="true"
    )
print(f"The following optimizations are enabled - {optimization_parameters}")
```

- change batch size to 8
- add this to the pipeline creation cell at the end of that cell

```
# set the pytorch and mlflow mode to mount
pipeline_object.jobs["summarization_pipeline"]["outputs"]["pytorch_model_folder"].mode = "mount"

pipeline_object.jobs["summarization_pipeline"]["outputs"]["mlflow_model_folder"].mode = "mount"
```

- Above will set the output for pytorch to mount mode
- other wise it will error at the end of fine tuning step

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel1.jpg "Architecture")

- compute

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel11.jpg "Architecture")

- Summary information

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel2.jpg "Architecture")

- metric

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel3.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel4.jpg "Architecture")

- GPU utilization

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel7.jpg "Architecture")

- GPU memory utilization

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel8.jpg "Architecture")

- GPU energy usage

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel19.jpg "Architecture")


## Second run

- using 2 A100 4 GPU computer
- Using SKU - Standard_NC48ads_A100_v4 (48 cores, 440 GB RAM, 128 GB disk)
- we are using 2 GPU VM to horizontally run the fine tuning
- same notebook as before
- make sure the changes in steps 3 and 5 are changed as above previous run
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel9.jpg "Architecture")

- compute

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel10.jpg "Architecture")

- Summary information

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel12.jpg "Architecture")

- Metrics

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel14.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel15.jpg "Architecture")

- GPU utilization

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel16.jpg "Architecture")

- GPU memory utilization

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel17.jpg "Architecture")

- GPU Energy usage

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel18.jpg "Architecture")

## Summary

- now both runs
- First one with Standard_NC48ads_A100_v4 (48 cores, 440 GB RAM, 128 GB disk)
- Second one is Standard_NC96ads_A100_v4 (96 cores, 880 GB RAM, 256 GB disk)

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/finetuneamlparallel20.jpg "Architecture")