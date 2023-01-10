# Predictive Maintenance using Azure Machine Learning AutoML and Inference using Managed Online endpoint

## Using AutoML to predict the failure of a machine

### Introduction

- Idea here is to predict the failure of a machine using AutoML
- AutoML can increase data science process productivity by automating time-consuming, iterative tasks
- Once the model is developed, we will deploy the model as a web service
- For deployment we are using Managed Online endpoint, Fully managed service that provides a REST endpoint for scoring
- We are using AutoMl to train the model
- Deploy the model using Managed Online endpoint
- Future article will cover to deploy to IoT Edge
- AutoML improves productivity by automating time-consuming, iterative tasks
- Cloud first PaaS approach

### Prerequisites

- Azure Subscription
- Azure machine learning Service
- Azure Storage
- Open source data set for NASA turbofan engine degradation simulation data set
- https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- data is also available in this repo data folder

## Code

### Create a workspace

- Create a workspace in Azure Machine Learning
- Upload the file to Azure storage or workspace

### Create a compute cluster

- Create a compute cluster in Azure Machine Learning
- name the cluster as cpu-cluster
- just one or two nodes are enough

### Create a notebook

- Create a notebook in Azure Machine Learning
- python kernel 3.10 with sdk v2
- import libraries

```
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
```

- import the data set

```
# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.identity import AzureCliCredential
from azure.ai.ml import automl, Input, MLClient

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import (
    classification,
    ClassificationPrimaryMetrics,
    ClassificationModels,
)
```

```
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()
ml_client = None
try:
    ml_client = MLClient.from_config(credential)
except Exception as ex:
    print(ex)
    # Enter details of your AzureML workspace
    subscription_id = "xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    resource_group = "rgname"
    workspace = "wkspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
```

```
workspace = ml_client.workspaces.get(name=ml_client.workspace_name)

subscription_id = ml_client.connections._subscription_id
resource_group = workspace.resource_group
workspace_name = ml_client.workspace_name

output = {}
output["Workspace"] = workspace_name
output["Subscription ID"] = subscription_id
output["Resource Group"] = resource_group
output["Location"] = workspace.location
output
```

- load the input

```
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_training_data_input = Input(
    type=AssetTypes.MLTABLE, path="./train/",
    description="Dataset for NASA Turbofan Training",
    tags={"source_type": "web", "source": "Kaggle ML Repo"},
    version="1.0.0",
)
my_training_data_test = Input(
    type=AssetTypes.MLTABLE, path="./test/",
    description="Dataset for NASA Turbofan Testing",
    tags={"source_type": "web", "source": "Kaggle ML Repo"},
    version="1.0.0",
)
```

- Set verion

```
rai_titanic_example_version_string = "1"
```

```
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

input_train_data = "nasaturbofan_train_csv"
input_test_data = "nasaturbofan_test_csv"

#input_train_data = "Data/train/"
#input_test_data = "Data/test/"


try:
    # Try getting data already registered in workspace
    my_training_data_input = ml_client.data.get(
        name=input_train_data, version=rai_titanic_example_version_string
    )
    my_training_data_test = ml_client.data.get(
        name=input_test_data, version=rai_titanic_example_version_string
    )
except Exception as e:
    my_training_data_input = Data(
        path="./train/",
        type=AssetTypes.MLTABLE,
        description="RAI NASA Turbofan training data",
        name=input_train_data,
        version=rai_titanic_example_version_string,
    )
    ml_client.data.create_or_update(my_training_data_input)

    my_training_data_test = Data(
        path="./test/",
        type=AssetTypes.MLTABLE,
        description="RAI NASA turbofan test data",
        name=input_test_data,
        version=rai_titanic_example_version_string,
    )
    ml_client.data.create_or_update(my_training_data_test)
```

- Job parameters

```
# General job parameters
compute_name = "cpu-cluster"
max_trials = 20
exp_name = "automlv2-NASATurbofan-experiment"
```

- Set path

```
train_data_path = "data/train/"
test_data_path = "data/test/"
```

- display sample data

```
import os
import pandas as pd
import mltable

tbl = mltable.load(train_data_path)
train_df: pd.DataFrame = tbl.to_pandas_dataframe()

# test dataset should have less than 5000 rows
test_df = mltable.load(test_data_path).to_pandas_dataframe()
assert len(test_df.index) <= 5000

display(train_df)
```

- now create the input dataset

```
my_training_data_input = Input(
    type=AssetTypes.MLTABLE, path="./data/train"
)
my_training_data_test = Input(
    type=AssetTypes.MLTABLE, path="./data/test"
)
my_training_data_validate = Input(
    type=AssetTypes.MLTABLE, path="./data/test"
)
```

- now automl configuration

```
# Create the AutoML classification job with the related factory-function.

classification_job = automl.classification(
    compute=compute_name,
    experiment_name=exp_name,
    training_data=my_training_data_input,
    target_column_name="broken",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    tags={"Training_Run": "Nasa Turbofan Pred Training"},
)

# Limits are all optional
classification_job.set_limits(
    timeout_minutes=600,
    trial_timeout_minutes=20,
    max_trials=max_trials,
    # max_concurrent_trials = 4,
    # max_cores_per_trial: -1,
    enable_early_termination=True,
)

# Training properties are optional
classification_job.set_training(
    blocked_training_algorithms=[ClassificationModels.LOGISTIC_REGRESSION],
    enable_onnx_compatible_models=True,
)
```

- Run submit the jon

```
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")
```

- wait for job to run

```
ml_client.jobs.stream(returned_job.name)
```

## inferencing code

- get mlflow URL

```
import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

print(MLFLOW_TRACKING_URI)
```

```
# Set the MLFLOW TRACKING URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
```

```
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
```

- lets pull the metrics

```
job_name = returned_job.name

# Example if providing an specific Job name/ID
# job_name = "b4e95546-0aa1-448e-9ad6-002e3207b4fc"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```

- best run

```
# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Found best child run id: ", best_child_run_id)

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)
```

- metrics

```
best_run.data.metrics
```

- download the model

```
import os

# Create local folder
local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
```

```
# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(
    best_run.info.run_id, "outputs", local_dir
)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))
```

```
# Show the contents of the MLFlow model folder
os.listdir("./artifact_downloads/outputs/mlflow-model")
```

- load the model

```
# import required libraries
# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
)
from azure.ai.ml.constants import ModelType
```

- Configure online endpoint

```
# Creating a unique endpoint name with current datetime to avoid conflicts
import datetime

online_endpoint_name = "nasaturbofan-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is a sample online endpoint for mlflow model",
    auth_mode="key",
    tags={"UseCase": "NASA Turbo fan prediction"},
)
ml_client.begin_create_or_update(endpoint)
```

```
model_name = "nasaturbofan-model"
model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/model.pkl",
    name=model_name,
    description="my sample mlflow model",
)

# for downloaded file
# model = Model(path="artifact_downloads/outputs/model.pkl", name=model_name)

registered_model = ml_client.models.create_or_update(model)
registered_model.id
```

- setup environment

```
env = Environment(
    name="automl-tabular-env",
    description="environment for automl inference",
    #image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
    conda_file="artifact_downloads/outputs/conda_env_v_1_0_0.yml",
)
```

```
code_configuration = CodeConfiguration(
    code="artifact_downloads/outputs/", scoring_script="scoring_file_v_2_0_0.py"
)
```

- create deployment

```
deployment = ManagedOnlineDeployment(
    name="nasaturbofan-deploy",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    environment=env,
    code_configuration=code_configuration,
    instance_type="Standard_DS2_V2",
    instance_count=1,
)
```

- status

```
ml_client.online_deployments.begin_create_or_update(deployment)
```

- Set the traffic

```
endpoint.traffic = {"nasaturbofan-deploy": 100}
#ml_client.begin_create_or_update(endpoint)
```

- test the REST API

```
# test the blue deployment with some sample data
ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="nasaturbofan-deploy",
    request_file="sample-nasaturbofan-data.json",
)
```

- output of the sample data below will be 1
- Entire JSON sample file is provided below in the article

- get URI data

```
# Get the details for online endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)
```

- getting logs

```
ml_client.online_deployments.get_logs(
    name=online_endpoint_name, endpoint_name=online_endpoint_name, lines=50
)
```

- delete the endpoint

```
ml_client.online_endpoints.begin_delete(name=online_endpoint_name)
```

- Sample json file
- sample-nasaturbofan-data.json

```
{
  "Inputs": {
    "data": [
      {
    "lifetime": 149,
    "unit_number": 1,
    "sensor10_max": 1.3,
    "sensor10_mean": 1.0900671140939597,
    "sensor10_min": 0.94,
    "sensor10_std": 0.12721813913402202,
    "sensor11_max": 48.09,
    "sensor11_mean": 43.02503355704698,
    "sensor11_min": 36.57,
    "sensor11_std": 3.2626300583294707,
    "sensor12_max": 521.83,
    "sensor12_mean": 259.8767785234899,
    "sensor12_min": 129.49,
    "sensor12_std": 141.68495100840693,
    "sensor13_max": 2388.43,
    "sensor13_mean": 2334.7675838926175,
    "sensor13_min": 2027.94,
    "sensor13_std": 128.112539555737,
    "sensor14_max": 8128.64,
    "sensor14_mean": 8047.956241610738,
    "sensor14_min": 7857.51,
    "sensor14_std": 81.280899508853,
    "sensor15_max": 10.9764,
    "sensor15_mean": 9.356339597315436,
    "sensor15_min": 8.3892,
    "sensor15_std": 0.739326323996306,
    "sensor16_max": 0.03,
    "sensor16_mean": 0.023489932885906038,
    "sensor16_min": 0.02,
    "sensor16_std": 0.004782594356544264,
    "sensor17_max": 397,
    "sensor17_mean": 347.6442953020134,
    "sensor17_min": 306,
    "sensor17_std": 28.129304193290917,
    "sensor18_max": 2388,
    "sensor18_mean": 2226.0805369127515,
    "sensor18_min": 1915,
    "sensor18_std": 145.0560244413751,
    "sensor19_max": 100,
    "sensor19_mean": 97.77489932885905,
    "sensor19_min": 84.93,
    "sensor19_std": 5.364169114956282,
    "sensor1_max": 518.67,
    "sensor1_mean": 471.63315436241606,
    "sensor1_min": 445,
    "sensor1_std": 27.183640153772533,
    "sensor20_max": 39.04,
    "sensor20_mean": 20.28489932885906,
    "sensor20_min": 10.36,
    "sensor20_std": 10.169620539404235,
    "sensor21_max": 23.3464,
    "sensor21_mean": 12.19033154362416,
    "sensor21_min": 6.2285,
    "sensor21_std": 6.07570852841947,
    "sensor2_max": 643.73,
    "sensor2_mean": 578.2204026845637,
    "sensor2_min": 536.25,
    "sensor2_std": 38.08074197407673,
    "sensor3_max": 1607.03,
    "sensor3_mean": 1417.8411409395972,
    "sensor3_min": 1256.76,
    "sensor3_std": 107.83043840522227,
    "sensor4_max": 1429.43,
    "sensor4_mean": 1204.3132214765099,
    "sensor4_min": 1040.99,
    "sensor4_std": 121.09415984683748,
    "sensor5_max": 14.62,
    "sensor5_mean": 7.8592617449664415,
    "sensor5_min": 3.91,
    "sensor5_std": 3.747488905913609,
    "sensor6_max": 21.61,
    "sensor6_mean": 11.348389261744966,
    "sensor6_min": 5.71,
    "sensor6_std": 5.619952999119863,
    "sensor7_max": 554.08,
    "sensor7_mean": 276.0636241610738,
    "sensor7_min": 137.57,
    "sensor7_std": 150.3105022549258,
    "sensor8_max": 2388.31,
    "sensor8_mean": 2225.9511409395977,
    "sensor8_min": 1915.02,
    "sensor8_std": 145.05832603497063,
    "sensor9_max": 9051.13,
    "sensor9_mean": 8494.444697986575,
    "sensor9_min": 7993.23,
    "sensor9_std": 337.5976689760314,
    "fold": 1
  }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}
```