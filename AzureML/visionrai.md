# Responsible AI with Image using Azure Machine Learning

## Responsible AI with Image

## Requirements

- Azure Subscription
- Azure Storage
- Azure Machine Learning
- Image dataset or use the sample dataset

## Steps

- First install the Azure Machine Learning SDK libraries

```
%pip install azure-identity
%pip install azure-ai-ml
%pip install mlflow
%pip install azureml-mlflow
```

```
%pip install pycocotools==2.0.6
%pip install simplification==0.6.11
%pip install scikit-image==0.19.3
```

- Set version

```
version_string = "0.0.11"
```

- Now configure training and cpu cluster for training and rai dashboard

```
# Compute cluster to run the AutoML training job
train_compute_name = "gpu-cluster"

# Compute cluster to visualize and interact with the RAI Dashboard
rai_compute_name = "cpu-cluster"
```

- Import the libraries

```
# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.automl import SearchSpace, ObjectDetectionPrimaryMetrics
from azure.ai.ml import automl, dsl
```

- Now setup the subscription and resource group

```
# Enter details of your AML workspace
subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
resource_group = "rgname"
workspace = "workspacename"
```

- Now create the client

```
# Handle to the workspace
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

try:
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )
except Exception:
    # If in compute instance we can get the config automatically
    from azureml.core import Workspace

    workspace = Workspace.from_config()
    workspace.write_config()
    ml_client = MLClient.from_config(
        credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
        logging_enable=True,
    )

print(ml_client)
```

- Now create the compute cluster

```
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

try:
    _ = ml_client.compute.get(train_compute_name)
    print("Found existing compute target.")
except ResourceNotFoundError:
    print("Creating a new compute target...")
    compute_config = AmlCompute(
        name=train_compute_name,
        type="amlcompute",
        size="Standard_NC6s_v3",
        idle_time_before_scale_down=120,
        min_instances=0,
        max_instances=4,
    )
    ml_client.begin_create_or_update(compute_config).result()
```

- Now create the rai compute cluster

```
from azure.ai.ml.entities import AmlCompute

all_compute_names = [x.name for x in ml_client.compute.list()]

if rai_compute_name in all_compute_names:
    print(f"Found existing compute: {rai_compute_name}")
else:
    rai_compute_config = AmlCompute(
        name=rai_compute_name,
        size="STANDARD_DS3_V2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=3600,
    )
    ml_client.compute.begin_create_or_update(rai_compute_config)
```

- Download sample dataset

```
import os
import urllib
from zipfile import ZipFile

# Change to a different location if you prefer
dataset_parent_dir = "./data"

# create data folder if it doesnt exist.
os.makedirs(dataset_parent_dir, exist_ok=True)

# download data
download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"

# Extract current dataset name from dataset url
dataset_name = os.path.split(download_url)[-1].split(".")[0]
# Get dataset path for later use
dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

# Get the data zip file path
data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

# Download the dataset
urllib.request.urlretrieve(download_url, filename=data_file)

# extract files
with ZipFile(data_file, "r") as zip:
    print("extracting files...")
    zip.extractall(path=dataset_parent_dir)
    print("done")
# delete zip file
os.remove(data_file)
```

- Display a sample image

```
from IPython.display import Image

sample_image = os.path.join(dataset_dir, "images", "31.jpg")
Image(filename=sample_image)
```

- create the dataset

```
# Uploading image files by creating a 'data asset URI FOLDER':

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml import Input

my_data = Data(
    path=dataset_dir,
    type=AssetTypes.URI_FOLDER,
    description="Fridge-items images Object detection",
    name="fridge-items-images-object-detection",
)

uri_folder_data_asset = ml_client.data.create_or_update(my_data)

print(uri_folder_data_asset)
print("")
print("Path to folder in Blob Storage:")
print(uri_folder_data_asset.path)
```

- Convert to JSONL file format

```
import sys

# use the jsonl-conversion files from automl examples folder
sys.path.insert(0, "jsonl-conversion/")
from base_jsonl_converter import write_json_lines
from voc_jsonl_converter import VOCJSONLConverter

base_url = os.path.join(uri_folder_data_asset.path, "images/")
converter = VOCJSONLConverter(base_url, os.path.join(dataset_dir, "annotations"))
jsonl_annotations = os.path.join(dataset_dir, "annotations_voc.jsonl")
write_json_lines(converter, jsonl_annotations)
```

- Create ML table

```
import os

# We'll copy each JSONL file within its related MLTable folder
training_mltable_path = os.path.join(dataset_parent_dir, "training-mltable-folder")
validation_mltable_path = os.path.join(dataset_parent_dir, "validation-mltable-folder")

# First, let's create the folders if they don't exist
os.makedirs(training_mltable_path, exist_ok=True)
os.makedirs(validation_mltable_path, exist_ok=True)

train_validation_ratio = 5

# Path to the training and validation files
train_annotations_file = os.path.join(training_mltable_path, "train_annotations.jsonl")
validation_annotations_file = os.path.join(
    validation_mltable_path, "validation_annotations.jsonl"
)

with open(jsonl_annotations, "r") as annot_f:
    json_lines = annot_f.readlines()

index = 0
with open(train_annotations_file, "w") as train_f:
    with open(validation_annotations_file, "w") as validation_f:
        for json_line in json_lines:
            if index % train_validation_ratio == 0:
                # validation annotation
                validation_f.write(json_line)
            else:
                # train annotation
                train_f.write(json_line)
            index += 1
```

- Creating the ML table file

```
def create_ml_table_file(filename):
    """Create ML Table definition"""

    return (
        "paths:\n"
        "  - file: ./{0}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info"
    ).format(filename)


def save_ml_table_file(output_path, mltable_file_contents):
    with open(os.path.join(output_path, "MLTable"), "w") as f:
        f.write(mltable_file_contents)


# Create and save train mltable
train_mltable_file_contents = create_ml_table_file(
    os.path.basename(train_annotations_file)
)
save_ml_table_file(training_mltable_path, train_mltable_file_contents)

# Save train and validation mltable
validation_mltable_file_contents = create_ml_table_file(
    os.path.basename(validation_annotations_file)
)
save_ml_table_file(validation_mltable_path, validation_mltable_file_contents)
```

- Set Input and validation dataset

```
# Training MLTable defined locally, with local data to be uploaded
my_training_data_input = Input(type=AssetTypes.MLTABLE, path=training_mltable_path)

# Validation MLTable defined locally, with local data to be uploaded
my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=validation_mltable_path)

# WITH REMOTE PATH: If available already in the cloud/workspace-blob-store
# my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/train")
# my_validation_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classificatio
```

- Set the target column

```
target_column_name = "label"
```

- Set experiment name

```
# general job parameters
exp_name = "dpv2-odfridge-automl-training"
```

- Setup AutoML config

```
# Create the AutoML job with the related factory-function.
image_object_detection_job = automl.image_object_detection(
    compute=train_compute_name,
    experiment_name=exp_name,
    training_data=my_training_data_input,
    validation_data=my_validation_data_input,
    target_column_name=target_column_name,
    primary_metric=ObjectDetectionPrimaryMetrics.MEAN_AVERAGE_PRECISION,
    tags={"data": "ODFridge", "model type": "AutoML"},
)

# Set limits
image_object_detection_job.set_limits(timeout_minutes=60)

# Pass the fixed settings or parameters
image_object_detection_job.set_training_parameters(
    model_name="fasterrcnn_resnet50_fpn",
    early_stopping=1,
    number_of_epochs=1,
    learning_rate=0.09,
)
```

- Submit the job

```
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(image_object_detection_job)

print(f"Created job: {returned_job}")
```

- Now wait for the job to complete

```
ml_client.jobs.stream(returned_job.name)
```

- Set the MLFlow tracking URI

```
import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

print(MLFLOW_TRACKING_URI)
```

- Set the tracking uri

```
# Set the MLFLOW TRACKING URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"\nCurrent tracking uri: {mlflow.get_tracking_uri()}")
```

- Invoke MLFlow

```
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
```

```
job_name = returned_job.name

# # Example if providing an specific Job name/ID
# job_name = "happy_yam_40fq53m7c2" #"ashy_net_gdd31zf2fq"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```

- Print the parent keys

```
# Print parent run tags. 'automl_best_child_run_id' tag should be there.
print(mlflow_parent_run.data.tags.keys())
```

- Get the best child run

```
# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print(f"Found best child run id: {best_child_run_id}")

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)
```

- Print metrics

```
import pandas as pd

# Access the results (such as Models, Artifacts, Metrics) of a previously completed AutoML Run.
pd.DataFrame(best_run.data.metrics, index=[0]).T
```

- Now get the model registered
- for that first create a folder to download the image

```
# Create local folder
import os

local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
```

- Now download the model

```
# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(
    best_run.info.run_id, "outputs", local_dir
)
print(f"Artifacts downloaded in: {local_path}")
print(f"Artifacts: {os.listdir(local_path)}")
```

- Display the content of the folder

```
import os

mlflow_model_dir = os.path.join(local_dir, "outputs", "mlflow-model")

# Show the contents of the MLFlow model folder
os.listdir(mlflow_model_dir)

# You should see a list of files such as the following:
# ['artifacts', 'conda.yaml', 'MLmodel', 'python_env.yaml', 'python_model.pkl', 'requirements.txt']
```

- Register the model

```
# import required libraries
from azure.ai.ml.entities import Model

model_name = "automl-fasterrcnn-odfridge-model"

model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/mlflow-model/",
    name=model_name,
    description="AutoML FasterRCNN model trained on OD Fridge Dataset",
    type=AssetTypes.MLFLOW_MODEL,
)

# for downloaded file
# model = Model(
#     path=mlflow_model_dir,
#     name=model_name,
#     description="AutoML FasterRCNN model trained on OD Fridge Dataset",
#     type=AssetTypes.MLFLOW_MODEL,
# )

registered_model = ml_client.models.create_or_update(model)
```

- Display the Registered model id

```
registered_model.id
expected_model_id = f"{registered_model.name}:{registered_model.version}"
azureml_model_id = f"azureml:{expected_model_id}"
```

- import the libraries

```
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

# from enum import Enum
import xml.etree.ElementTree as ET
import pandas as pd
from zipfile import ZipFile

from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
```

- invoke RAI ML client

```
registry_name = "azureml"
credential = DefaultAzureCredential()

ml_client_registry = MLClient(
    credential=credential,
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    registry_name=registry_name,
)

rai_vision_insights_component = ml_client_registry.components.get(
    name="rai_vision_insights", version=version_string
)
```

- Set the RAI configuation

```
import json
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes


@dsl.pipeline(
    compute=rai_compute_name,
    description="Example RAI computation on Fridge data with AutoML FasterRCNN Object Detection model",
    experiment_name=f"RAI_Fridge_Example_RAIInsights_Computation_{expected_model_id}",
)
def rai_fridge_object_detection_pipeline(target_column_name, test_data, classes):
    # Initiate the RAIInsights
    rai_image_job = rai_vision_insights_component(
        task_type="object_detection",
        model_info=expected_model_id,
        model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
        test_dataset=test_data,
        target_column_name=target_column_name,
        classes=classes,
        model_type="pyfunc",
        precompute_explanation=True,
        dataset_type="private",
        enable_error_analysis=False,
        maximum_rows_for_test_dataset=5000,
        num_masks=300,
        mask_res=4,
    )
    rai_image_job.set_limits(timeout=24000)

    rai_image_job.outputs.dashboard.mode = "upload"
    rai_image_job.outputs.ux_json.mode = "upload"

    return {
        "dashboard": rai_image_job.outputs.dashboard,
        "ux_json": rai_image_job.outputs.ux_json,
    }
```

- Create the RAI pipeline

```
import uuid
from azure.ai.ml import Output

insights_pipeline_job = rai_fridge_object_detection_pipeline(
    target_column_name=target_column_name,
    test_data=fridge_test_mltable,
    classes='["can", "carton", "milk_bottle", "water_bottle"]',  # Ensure the class order matches that of used while creating the jsonl file for the test data
)

rand_path = str(uuid.uuid4())
insights_pipeline_job.outputs.dashboard = Output(
    path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
    mode="upload",
    type="uri_folder",
)
insights_pipeline_job.outputs.ux_json = Output(
    path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/ux_json/",
    mode="upload",
    type="uri_folder",
)
```

- Submit the job

```
# Submit the RAI Vision Insights job
returned_job = ml_client.jobs.create_or_update(insights_pipeline_job)

print(f"Created job: {returned_job}")
```

- Wait for the job to complete

```
ml_client.jobs.stream(returned_job.name)
```

- now display the url for ouptut

```
sub_id = ml_client._operation_scope.subscription_id
rg_name = ml_client._operation_scope.resource_group_name
ws_name = ml_client.workspace_name

expected_uri = f"https://ml.azure.com/model/{expected_model_id}/model_analysis?wsid=/subscriptions/{sub_id}/resourcegroups/{rg_name}/workspaces/{ws_name}"

print(f"Please visit {expected_uri} to see your analysis")
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai1.jpg "Output Episodes")

- Training run

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai2.jpg "Output Episodes")

- Responsible AI dashbaord

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai3.jpg "Output Episodes")

- go into the RAI job

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai4.jpg "Output Episodes")

- Click Open to take you to Responsible AI dashboard
- Set a compute instance for Responsible AI dashboard

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai5.jpg "Output Episodes")

- Let's look into RAI Explorer to show Success and failed predictions

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai6.jpg "Output Episodes")

- There is a settings to change cohorts, if you have created multiple ones

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai7.jpg "Output Episodes")

- Now lets see the data set cohorts

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai8.jpg "Output Episodes")

- Check the feature cohorts

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai9.jpg "Output Episodes")

- Next to Data Analysis
- Table View

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai10.jpg "Output Episodes")

- Chart View
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai11.jpg "Output Episodes")

- Plot the predicted label

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/visionrai12.jpg "Output Episodes")