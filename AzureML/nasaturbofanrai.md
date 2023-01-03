# Responsible AI on NASA Turbofan Engine Degradation Dataset

## End to End Train model and perform Responsible AI on NASA Turbofan Engine Degradation Dataset

### Introduction

- Using NASA Turbofan Engine Degradation Dataset, we will train a model to predict Remaining Useful Life (RUL) of an engine.
- Goal is to show how to train the model using automl and perform responsible AI on the model.
- Get the best run model and perform responsible AI on the model.
- download data from - https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- Kaggle link - https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## Data Engineering Code

### Import Libraries

```
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
```

- Data load function
- Combine multiple files
- Create columns names for the data set for training and testing

```
def load_data(index="FD004"):
    if type(index) == str:
        assert index in ["FD001", "FD002", "FD003", "FD004"]
    elif type(index) == int:
        assert index in [0, 1, 2, 3]
        index = f'FD00{index+1}'

    print("-----------------")
    print(f" Data Set: {index} ")
    print("-----------------")
    if index == "FD001":
        print("Train trjectories: 100")
        print("Test trajectories: 100")
        print("Conditions: ONE (Sea Level)")
        print("Fault Modes: ONE (HPC Degradation)\n")
    if index == "FD002":
        print("Train trjectories: 260")
        print("Test trajectories: 259")
        print("Conditions: SIX")
        print("Fault Modes: ONE (HPC Degradation)\n")
    if index == "FD003":
        print("Train trjectories: 100")
        print("Test trajectories: 100")
        print("Conditions: ONE (Sea Level)")
        print("Fault Modes: TWO (HPC Degradation, Fan Degradation)\n")
    if index == "FD004":
        print("Train trjectories: 248")
        print("Test trajectories: 249")
        print("Conditions: SIX")
        print("Fault Modes: TWO (HPC Degradation, Fan Degradation)\n")

    train_set = np.loadtxt(f"train_{index}.txt")
    test_set  = np.loadtxt(f"test_{index}.txt")

    col_names = ["unit_number", "time"]
    col_names += [f"operation{i}" for i in range(1, 4)]
    col_names += [f"sensor{i}" for i in range(1, 22)]
    train_set = pd.DataFrame(train_set, columns=col_names)
    test_set  = pd.DataFrame(test_set, columns=col_names)
    labels = np.loadtxt(f"RUL_{index}.txt")

    def set_dtype(df):
        return df.astype({"unit_number": np.int64, "time": np.int64})

    train_set = set_dtype(train_set)
    test_set  = set_dtype(test_set)

    return train_set, test_set, labels
```

- load data 

```
train_set, test_set, labels = load_data(index=3)
```

- Function def to check failure

```
def run_to_failure_aux(df, lifetime, unit_number):

    assert lifetime <= df.shape[0]
    broken = 0 if lifetime < df.shape[0] else 1
    sample = pd.DataFrame(
        {'lifetime': lifetime, 'broken': broken, 'unit_number': unit_number}, index=[0])

    sensors = df.loc[:, df.columns.str.contains('sensor')]
    num_features = sensors.iloc[:lifetime].agg(['min', 'max', 'mean', 'std'])
    num_features = num_features.unstack().reset_index()
    num_features['feature'] = num_features.level_0.str.cat(
        num_features.level_1, sep='_')
    num_features = num_features.pivot_table(columns='feature', values=0)

    return pd.concat([sample, num_features], axis=1)
```

- censor augumentation function

```
def censoring_augmentation(raw_data, n_samples=10, seed=123):

    np.random.seed(seed)
    datasets = [g for _, g in raw_data.groupby('unit_number')]
    timeseries = raw_data.groupby('unit_number').size()
    samples = []
    pbar = tqdm.tqdm(total=n_samples, desc='augmentation')

    while len(samples) < n_samples:
        # draw a machine
        unit_number = np.random.randint(timeseries.shape[0])
        censor_timing = np.random.randint(timeseries.iloc[unit_number])
        sample = run_to_failure_aux(datasets[unit_number], censor_timing, unit_number)
        samples.append(sample)
        pbar.update(1)

    return pd.concat(samples).reset_index(drop=True).fillna(0)
```

- Generate run to failure data

```
def generate_run_to_failure(df, health_censor_aug=0, seed=123):

    samples = []
    for unit_id, timeseries in tqdm.tqdm(df.groupby('unit_number'), desc='RUL'):
        samples.append(run_to_failure_aux(timeseries, timeseries.shape[0], unit_id))

    samples = pd.concat(samples)

    if health_censor_aug > 0:
        aug_samples = censoring_augmentation(
            df, n_samples=health_censor_aug, seed=seed)
        return pd.concat([samples, aug_samples]).reset_index(drop=True)
    else:
        return samples.reset_index(drop=True)
```

- load the data now

```
dataset = generate_run_to_failure(train_set,
    health_censor_aug=train_set.unit_number.nunique() * 3)

dataset.sample(10).sort_index()
```

```
dataset.unit_number.hist(bins=50)
```

```
dataset.broken.hist()
```

- now clean out the file

```
def leave_one_out(target='run-to-failure',
                  health_censor_aug=1000, seed=123,
                  input_fn=None, output_fn=None):

    if input_fn is not None:
        subsets = pd.read_csv(input_fn)

    else:
        subsets = []
        for index in range(4):
            raw_data = load_data(index=index)[0]
            raw_data = raw_data.assign(machine_id=index)

            if target == 'run-to-failure':
                subset = generate_run_to_failure(raw_data, health_censor_aug, seed)
                subset = subset.assign(fold=index)
                subsets.append(subset)

            elif target == 'time-to-failure':
                raise NotImplementedError

            else:
                raise ValueError

        subsets = pd.concat(subsets).reset_index(drop=True)

    if output_fn is not None:
        subsets.to_csv(output_fn, index=False)

    # List of tuples: (train_data, test_data)
    train_test_sets = [(
        subsets[subsets.fold != i].reset_index(drop=True),
        subsets[subsets.fold == i].reset_index(drop=True)) for i in range(4)]

    return train_test_sets
```

- rebuild clean dataset

```
dataset = leave_one_out()
```

- split train and test

```
train_set, test_set = dataset[0]
train_set
```

```
train_set.lifetime.hist()
```

- Generate validation dataset

```
def generate_validation_sets(method='leave-one-out', n_splits=5, seed=123, outdir=None):
    validation_sets = []

    if method == 'kfold':
        raise NotImplementedError

    elif method == 'leave-one-out':
        validation_sets = leave_one_out(target='run-to-failure',
                                        health_censor_aug=1000,
                                        seed=seed)

        if outdir is not None:
            for i, (train_data, test_data) in enumerate(validation_sets):
                train_data.to_csv(outdir + f'/train_{i}.csv.gz', index=False)
                test_data.to_csv(outdir + f'/test_{i}.csv.gz', index=False)        

    return validation_sets
```

```
val = generate_validation_sets()
```

```
val[0][0].broken.hist()
plt.show()
val[0][1].broken.hist()
```

```
len(val)
```

## Training Code now

- import libraries

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

- create a workspace

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
    subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    resource_group = "rgname"
    workspace = "amlwkspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
```

- Display the workspace

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

- Now save the output for traning
- Create 2 folders called train and test
- Create a MLTable file in each folder

- MLTable in train
  
```
# MLTable definition file

paths:
  - file: ./train_set.csv
transformations:
  - read_delimited:
        delimiter: ','
        encoding: 'ascii'

- MLTable in test

```
# MLTable definition file

paths:
  - file: ./test_set.csv
transformations:
  - read_delimited:
        delimiter: ','
        encoding: 'ascii'
```

- Now lets save the train and test csv files

```
train_set.to_csv("train/train_set.csv",index=False, header=True)
```

```
train_set.to_csv("test/test_set.csv",index=False, header=True)
```

- Now create dataset definition for automl training

```
my_training_data_input = Input(
    type=AssetTypes.MLTABLE, path="./train/"
)
my_training_data_test = Input(
    type=AssetTypes.MLTABLE, path="./test/"
)
```

- Set the version string 

```
rai_titanic_example_version_string = "1"
```

- Now create a dataset and register in the workspace UI

``` 
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

input_train_data = "nasaturbofan_train_csv"
input_test_data = "nasaturbofan_test_csv"

#input_train_data = "Data/train/"
#input_test_data = "Data/test/"


try:
    # Try getting data already registered in workspace
    train_data = ml_client.data.get(
        name=input_train_data, version=rai_titanic_example_version_string
    )
    test_data = ml_client.data.get(
        name=input_test_data, version=rai_titanic_example_version_string
    )
except Exception as e:
    train_data = Data(
        path="./train/",
        type=AssetTypes.MLTABLE,
        description="RAI NASA Turbofan training data",
        name=input_train_data,
        version=rai_titanic_example_version_string,
    )
    ml_client.data.create_or_update(train_data)

    test_data = Data(
        path="./test/",
        type=AssetTypes.MLTABLE,
        description="RAI NASA turbofan test data",
        name=input_test_data,
        version=rai_titanic_example_version_string,
    )
    ml_client.data.create_or_update(test_data)
```

- Setup job parameters

```
# General job parameters
compute_name = "cpu-cluster"
max_trials = 5
exp_name = "automlv2-NASATurbofan-experiment"
```

- Setup the automl job

```
classification_job = automl.classification(
    compute=compute_name,
    experiment_name=exp_name,
    training_data=my_training_data_input,
    target_column_name="broken",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    tags={"my_custom_tag": "NASA Turbofan Training"},
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

- Submit the job

```
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")
```

- Now lets check the status of the job

```
ml_client.jobs.stream(returned_job.name)
```

- now load mlflow URI

```
import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

print(MLFLOW_TRACKING_URI)
```

- Set the tracking URI

```
# Set the MLFLOW TRACKING URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
```

- Invoke MLFlow UI

```
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
```

- Get the run information

```
job_name = returned_job.name

# Example if providing an specific Job name/ID
# job_name = "b4e95546-0aa1-448e-9ad6-002e3207b4fc"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```

- Get the best child run

```
# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Found best child run id: ", best_child_run_id)

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)
```

- print the best run metrics

```
best_run.data
```

- get best run run name

```
print(best_run.data.tags['mlflow.runName'])
```

- load the best model

```
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

job_name = ""

run_model = Model(
    path=f"azureml://jobs/{best_run.data.tags['mlflow.rootRunId']}/outputs/artifacts/paths/model/",
    name="nasaturbofan_version1",
    description="Model created from run for NASA Turbofan.",
    type=AssetTypes.MLFLOW_MODEL,
)
```

- print the model name

```
run_model.name
```

- Register the model

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
os.listdir("./artifact_downloads/outputs/mlflow-model")
```

- Model name

```
expected_model_id = f"nasaturbofan_version1:2"
modelname = "nasaturbofan_version1"
azureml_model_id = f"azureml:{expected_model_id}"
```

- Register the model

```
import os

model_local_path = os.path.abspath("./artifact_downloads/outputs/mlflow-model")
mlflow.register_model(f"file://{model_local_path}", modelname)
```

```
import mlflow
import mlflow.sklearn
```

## Responsible AI (RAI) and Explainability

- Setup the workspace information

```
# Enter details of your AML workspace
subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
resource_group = "rgname"
workspace = "amlwkname"
```

- Load the registry where RAI components are registered

```
# Get handle to azureml registry for the RAI built in components
registry_name = "azureml"
#registry_name = "mlopswk"
ml_client_registry = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    registry_name=registry_name,
)
print(ml_client_registry)
```

 - Code for registering in the registry is available here - https://github.com/Azure/azureml-examples/blob/main/sdk/python/responsible-ai/responsibleaidashboard-housing-decision-making/responsibleaidashboard-housing-decision-making.ipynb
 - Now load the RAI components
 - We are calling Dashboard, Score card and Counterfactual components

 ```
 label = "latest"

rai_constructor_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_constructor", label=label, 
)

# We get latest version and use the same version for all components
version = rai_constructor_component.version
print("The current version of RAI built-in components is: " + version)

rai_explanation_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_explanation", version=version
)

rai_causal_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_causal", version=version
)

rai_counterfactual_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_counterfactual", version=version
)

rai_erroranalysis_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_erroranalysis", version=version
)

rai_gather_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_gather", version=version
)

rai_scorecard_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_score_card", version=version
)
```

- Setup up score card info

```
import json

score_card_config_dict = {
    "Model": {
        "ModelName": "NSAS Turbofan classification",
        "ModelType": "Classification",
        "ModelSummary": "",
    },
    "Metrics": {"accuracy_score": {"threshold": ">=0.7"}, "precision_score": {}},
}

score_card_config_filename = "rai_nasaturbofan_classification_score_card_config.json"

with open(score_card_config_filename, "w") as f:
    json.dump(score_card_config_dict, f)

score_card_config_path = Input(
    type="uri_file", path=score_card_config_filename, mode="download"
)
```

```
import time

model_name_suffix = int(time.time())
model_name = "rai_nasaturbofan_classifier"
```

- Setup configuration for RAI analysis
- Set the targe column name and categorical features

```
target_column_name = "broken"
categorical_features = []
```

- Setup Responsible AI pipeline

```
import json
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import dsl, Input

classes_in_target = json.dumps(["Less than median", "More than median"])
treatment_features = json.dumps(
    ["OverallCond", "OverallQual", "Fireplaces", "GarageCars", "ScreenPorch"]
)


@dsl.pipeline(
    compute=compute_name,
    description="Example RAI computation on housing data",
    experiment_name=f"RAI_Housing_Example_RAIInsights_Computation_{model_name_suffix}",
)
def rai_classification_pipeline(
    target_column_name,
    train_data,
    test_data,
    score_card_config_path,
):
    # Initiate the RAIInsights
    create_rai_job = rai_constructor_component(
        title="RAI Nasa Turbofan Dashboard Example",
        task_type="classification",
        model_info=expected_model_id,
        model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
        train_dataset=train_data,
        test_dataset=test_data,
        target_column_name=target_column_name,
        categorical_column_names=json.dumps(categorical_features),
        classes=classes_in_target,
    )
    create_rai_job.set_limits(timeout=120)

    # Add an explanation
    explain_job = rai_explanation_component(
        comment="Explanation for the NASA Turbofan dataset",
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    explain_job.set_limits(timeout=120)

    # Add causal analysis
    causal_job = rai_causal_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        treatment_features=treatment_features,
    )
    causal_job.set_limits(timeout=120)

    # Add counterfactual analysis
    counterfactual_job = rai_counterfactual_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        total_cfs=10,
        desired_class="opposite",
    )
    counterfactual_job.set_limits(timeout=600)

    # Add error analysis
    erroranalysis_job = rai_erroranalysis_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    erroranalysis_job.set_limits(timeout=120)

    # Combine everything
    rai_gather_job = rai_gather_component(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_1=explain_job.outputs.explanation,
        insight_2=causal_job.outputs.causal,
        insight_3=counterfactual_job.outputs.counterfactual,
        insight_4=erroranalysis_job.outputs.error_analysis,
    )
    rai_gather_job.set_limits(timeout=120)

    rai_gather_job.outputs.dashboard.mode = "upload"
    rai_gather_job.outputs.ux_json.mode = "upload"

    # Generate score card in pdf format for a summary report on model performance,
    # and observe distrbution of error between prediction vs ground truth.
    rai_scorecard_job = rai_scorecard_component(
        dashboard=rai_gather_job.outputs.dashboard,
        pdf_generation_config=score_card_config_path,
    )

    return {
        "dashboard": rai_gather_job.outputs.dashboard,
        "ux_json": rai_gather_job.outputs.ux_json,
        "scorecard": rai_scorecard_job.outputs.scorecard,
    }
```

- Setup the pipeline

```
import uuid
from azure.ai.ml import Output

# Pipeline to construct the RAI Insights
insights_pipeline_job = rai_classification_pipeline(
    target_column_name=target_column_name,
    train_data=my_training_data_input,
    test_data=my_training_data_test,
    score_card_config_path=score_card_config_path,
)

# Set the output path for the dashboard
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
insights_pipeline_job.outputs.scorecard = Output(
    path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/scorecard/",
    mode="upload",
    type="uri_folder",
)
```

- Submit the pipeline

```
nasaturbofanrai_job = ml_client.jobs.create_or_update(insights_pipeline_job)

print(f"Created job: {nasaturbofanrai_job}")
```

- Wait for job to complete

```
ml_client.jobs.stream(nasaturbofanrai_job.name)
```

- Download the dashboard

```
target_directory = "."

ml_client.jobs.download(
    nasaturbofanrai_job.name, download_path=target_directory, output_name="scorecard"
)
```