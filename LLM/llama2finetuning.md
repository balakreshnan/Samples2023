# LLama 2 Fine tuning using Azure Machine learning in parallel

## Llama 2 7,13, 70 Billion parameters fine tuning

## Code

- Import libraries

```
%pip install azure-ai-ml
%pip install azure-identity
%pip install datasets==2.9.0
%pip install mlflow
%pip install azureml-mlflow
```

```
%pip install datasets==2.9.0
%pip install py7zr
```

- Log into the Azure Machine learning workspace

```
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)
from azure.ai.ml.entities import AmlCompute
import time

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:
    workspace_ml_client = MLClient(
        credential,
        subscription_id="<SUBSCRIPTION_ID>",
        resource_group_name="<RESOURCE_GROUP>",
        workspace_name="<WORKSPACE_NAME>",
    )

# the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"
registry_ml_client = MLClient(credential, registry_name="azureml")
registry_ml_client_meta = MLClient(credential, registry_name="azureml-meta")

experiment_name = "text-generation-samsum"

# generating a unique timestamp that can be used for names and versions that need to be unique
timestamp = str(int(time.time()))
```

- now define the model

```
#model_name = "Llama-2-7b"
model_name = "Llama-2-70b"
#model_name = "Llama-2-13b"
foundation_model = registry_ml_client_meta.models.get(model_name, label="latest")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)
```

- Adding lower grade GPU

```
import ast

if "computes_allow_list" in foundation_model.tags:
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["computes_allow_list"]
    )  # convert string to python list
    computes_allow_list.append("Standard_NC48ads_A100_v4")
    computes_allow_list.append("Standard_NC96ads_A100_v4") 
    print(f"Please create a compute from the above list - {computes_allow_list}")
else:
    computes_allow_list = None
    print("Computes allow list is not part of model tags")
```

- Setup GPU Compute

```
# If you have a specific compute size to work with change it here. By default we use the 8 x V100 compute from the above list
compute_cluster_size = "Standard_NC96ads_A100_v4"

# If you already have a gpu cluster, mention it here. Else will create a new one with the name 'gpu-cluster-big'
#compute_cluster = "gpu-cluster-big"
compute_cluster = "gpu-cluster-large2"

try:
    compute = workspace_ml_client.compute.get(compute_cluster)
    print("The compute cluster already exists! Reusing it for the current run")
except Exception as ex:
    print(
        f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {compute_cluster_size}!"
    )
    try:
        print("Attempt #1 - Trying to create a dedicated compute")
        compute = AmlCompute(
            name=compute_cluster,
            size=compute_cluster_size,
            tier="Dedicated",
            max_instances=2,  # For multi node training set this to an integer value more than 1
        )
        workspace_ml_client.compute.begin_create_or_update(compute).wait()
    except Exception as e:
        try:
            print(
                "Attempt #2 - Trying to create a low priority compute. Since this is a low priority compute, the job could get pre-empted before completion."
            )
            compute = AmlCompute(
                name=compute_cluster,
                size=compute_cluster_size,
                tier="LowPriority",
                max_instances=2,  # For multi node training set this to an integer value more than 1
            )
            workspace_ml_client.compute.begin_create_or_update(compute).wait()
        except Exception as e:
            print(e)
            raise ValueError(
                f"WARNING! Compute size {compute_cluster_size} not available in workspace"
            )


# Sanity check on the created compute
#compute = workspace_ml_client.compute.get(compute_cluster)
#if compute.provisioning_state.lower() == "failed":
#    raise ValueError(
#        f"Provisioning failed, Compute '{compute_cluster}' is in failed state. "
#        f"please try creating a different compute"
#    )

#if computes_allow_list is not None:
#    computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
#    if compute.size.lower() not in computes_allow_list_lower_case:
#        raise ValueError(
#            f"VM size {compute.size} is not in the allow-listed computes for finetuning"
#        )
#else:
#    # Computes with K80 GPUs are not supported
#    unsupported_gpu_vm_list = [
#        "standard_nc6",
#        "standard_nc12",
#        "standard_nc24",
#        "standard_nc24r",
#    ]
#    if compute.size.lower() in unsupported_gpu_vm_list:
#        raise ValueError(
#            f"VM size {compute.size} is currently not supported for finetuning"
#        )


# This is the number of GPUs in a single node of the selected 'vm_size' compute.
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
gpu_count_found = False
workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
available_sku_sizes = []
for compute_sku in workspace_compute_sku_list:
    available_sku_sizes.append(compute_sku.name)
    if compute_sku.name.lower() == compute.size.lower():
        gpus_per_node = compute_sku.gpus
        gpu_count_found = True
        print(compute_sku.name.lower())
# if gpu_count_found not found, then print an error
if gpu_count_found:
    print(f"Number of GPU's in compute {compute.size}: {gpus_per_node}")
else:
    raise ValueError(
        f"Number of GPU's in compute {compute.size} not found. Available skus are: {available_sku_sizes}."
        f"This should not happen. Please check the selected compute cluster: {compute_cluster} and try again."
    )
```

- Prepare the dataset

```
# download the dataset using the helper script. This needs datasets library: https://pypi.org/project/datasets/
import os

exit_status = os.system("python ./download-dataset.py --download_dir samsum-dataset")
if exit_status != 0:
    raise Exception("Error downloading dataset")
```

- check the data

```
# load the ./samsum-dataset/train.jsonl file into a pandas dataframe and show the first 5 rows
import pandas as pd

pd.set_option(
    "display.max_colwidth", 0
)  # set the max column width to 0 to display the full text
df = pd.read_json("./samsum-dataset/train.jsonl", lines=True)
df.head()
```

- pre process dataset

```
# create a function to preprocess the dataset in desired format


def get_preprocessed_samsum(df):
    prompt = f"Summarize this dialog:\n{{}}\n---\nSummary:\n"

    df["text"] = df["dialogue"].map(prompt.format)
    df = df.drop(columns=["dialogue", "id"])
    df = df[["text", "summary"]]

    return df
```

- Split the train

```
# load test.jsonl, train.jsonl and validation.jsonl form the ./samsum-dataset folder into pandas dataframes
test_df = pd.read_json("./samsum-dataset/test.jsonl", lines=True)
train_df = pd.read_json("./samsum-dataset/train.jsonl", lines=True)
validation_df = pd.read_json("./samsum-dataset/validation.jsonl", lines=True)
# map the train, validation and test dataframes to preprocess function
train_df = get_preprocessed_samsum(train_df)
validation_df = get_preprocessed_samsum(validation_df)
test_df = get_preprocessed_samsum(test_df)
# show the first 5 rows of the train dataframe
train_df.head()
```

- Split to jsonl

```
# save 10% of the rows from the train, validation and test dataframes into files with small_ prefix in the ./samsum-dataset folder
frac = 1
train_df.sample(frac=frac).to_json(
    "./samsum-dataset/small_train.jsonl", orient="records", lines=True
)
validation_df.sample(frac=frac).to_json(
    "./samsum-dataset/small_validation.jsonl", orient="records", lines=True
)
test_df.sample(frac=frac).to_json(
    "./samsum-dataset/small_test.jsonl", orient="records", lines=True
)
```

- Get the experiment ready
- For llama2-7B with A100 4 GPU machine, max batch will be 32
- For llama2-7B with A100 4 GPU machine, max batch will be 8
- For Llama2-70B with A100 4 GPU machine, max batch will be 1
- i was unable to test with 8 GPU machine due to capacity constraints
- was able to test with 2 VM's to 13VM's with help of my customer.

```
# Training parameters
training_parameters = dict(
    num_train_epochs=3,
    #per_device_train_batch_size=32,
    #per_device_eval_batch_size=32,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
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

- now setup the pipeline

```
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input

# fetch the pipeline component
pipeline_component_func = registry_ml_client.components.get(
    name="text_generation_pipeline", label="latest"
)


# define the pipeline job
@pipeline()
def create_pipeline():
    text_generation_pipeline = pipeline_component_func(
        # specify the foundation model available in the azureml system registry id identified in step #3
        mlflow_model_path=foundation_model.id,
        # huggingface_id = 'meta-llama/Llama-2-7b', # if you want to use a huggingface model, uncomment this line and comment the above line
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        compute_model_evaluation=compute_cluster,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./samsum-dataset/small_train.jsonl"
        ),
        validation_file_path=Input(
            type="uri_file", path="./samsum-dataset/small_validation.jsonl"
        ),
        test_file_path=Input(type="uri_file", path="./samsum-dataset/small_test.jsonl"),
        evaluation_config=Input(type="uri_file", path="./text-generation-config.json"),
        # The following parameters map to the dataset fields
        text_key="text",
        ground_truth_key="summary",
        num_nodes_finetune=2,
        # Training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,  # set to the number of GPUs available in the compute
        **training_parameters,
        **optimization_parameters
    )
    return {
        # map the output of the fine tuning job to the output of pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": text_generation_pipeline.outputs.mlflow_model_folder
    }


pipeline_object = create_pipeline()

# don't use cached results from previous jobs
pipeline_object.settings.force_rerun = True

# set continue on step failure to False
pipeline_object.settings.continue_on_step_failure = False

# set the pytorch and mlflow mode to mount

pipeline_object.jobs["text_generation_pipeline"]["outputs"]["pytorch_model_folder"].mode = "mount"

pipeline_object.jobs["text_generation_pipeline"]["outputs"]["mlflow_model_folder"].mode = "mount"
```

- submit the experiment

```
# submit the pipeline job
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)
# wait for the pipeline job to complete
workspace_ml_client.jobs.stream(pipeline_job.name)
```

- if it fails go to designer of the experiment and go next level graph clone and delete the validation step and save and submit again
- for deep speed. there will be a separated JSON file with config provided.