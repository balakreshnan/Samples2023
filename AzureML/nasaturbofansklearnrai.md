# Predictive Maintenance - Responsible AI on NASA Turbofan Engine Degradation Dataset - Using sklearn

## End to End Train model and perform Responsible AI on NASA Turbofan Engine Degradation Dataset

### Introduction

- Using NASA Turbofan Engine Degradation Dataset, we will train a model to predict Remaining Useful Life (RUL) of an engine.
- Goal is to show how to train the model using automl and perform responsible AI on the model.
- Get the best run model and perform responsible AI on the model.
- download data from - https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- Kaggle link - https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- using sci-kit learn for training the model

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
    subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
    resource_group = "rgname"
    workspace = "azuremlworkspace"
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

- Create mltable based Table dataset to be used for both Training and RAI

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

- Configure version numbers

```
rai_titanic_example_version_string = "1"
```

- Create data if it doesn't exist

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

- Create component directories

```
import os

train_src_dir = "./components/train"
os.makedirs(train_src_dir, exist_ok=True)
```

- Now lets create the training code and file
- walk through each line to see how we are using data form mltable
- Clean up the data
- then train the model
- Save the model
- Register the model
- Note we only use one SKlearn model
- Enable mlflow logging
- RAI Toolbox at the time of implementation only works with sklearn and mlflow enabled training

```
%%writefile {train_src_dir}/train.py
import argparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import os
import shutil
import tempfile
import pandas as pd
import mlflow
import mltable
import pandas as pd 
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()
    
    #current_experiment = Run.get_context().experiment
    #tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    #print("tracking_uri: {0}".format(tracking_uri))
    #mlflow.set_tracking_uri(tracking_uri)
    #mlflow.set_experiment(current_experiment.name)

    # paths are mounted as folder, therefore, we are selecting the file from folder
    #train_df = pd.read_csv(select_first_file(args.train_data))
    tbl = mltable.load(args.train_data)
    train_df = tbl.to_pandas_dataframe()
    train_df.columns = ['lifetime','broken','unit_number','sensor10_max','sensor10_mean','sensor10_min','sensor10_std','sensor11_max','sensor11_mean','sensor11_min','sensor11_std','sensor12_max','sensor12_mean','sensor12_min','sensor12_std','sensor13_max','sensor13_mean','sensor13_min','sensor13_std','sensor14_max','sensor14_mean','sensor14_min','sensor14_std','sensor15_max','sensor15_mean','sensor15_min','sensor15_std','sensor16_max','sensor16_mean','sensor16_min','sensor16_std','sensor17_max','sensor17_mean','sensor17_min','sensor17_std','sensor18_max','sensor18_mean','sensor18_min','sensor18_std','sensor19_max','sensor19_mean','sensor19_min','sensor19_std','sensor1_max','sensor1_mean','sensor1_min','sensor1_std','sensor20_max','sensor20_mean','sensor20_min','sensor20_std','sensor21_max','sensor21_mean','sensor21_min','sensor21_std','sensor2_max','sensor2_mean','sensor2_min','sensor2_std','sensor3_max','sensor3_mean','sensor3_min','sensor3_std','sensor4_max','sensor4_mean','sensor4_min','sensor4_std','sensor5_max','sensor5_mean','sensor5_min','sensor5_std','sensor6_max','sensor6_mean','sensor6_min','sensor6_std','sensor7_max','sensor7_mean','sensor7_min','sensor7_std','sensor8_max','sensor8_mean','sensor8_min','sensor8_std','sensor9_max','sensor9_mean','sensor9_min','sensor9_std','fold']
    train_df = train_df.iloc[1: , :]
    print(train_df.columns)
    print(train_df.head())
    train_df.fold.fillna(0, inplace=True)

    # Extracting the label column
    y_train = train_df.pop("broken")

    # convert the dataframe values to array
    X_train = train_df.values

    # paths are mounted as folder, therefore, we are selecting the file from folder
    #test_df = pd.read_csv(select_first_file(args.test_data))
    tbl = mltable.load(args.test_data)
    test_df = tbl.to_pandas_dataframe()
    test_df.columns = ['lifetime','broken','unit_number','sensor10_max','sensor10_mean','sensor10_min','sensor10_std','sensor11_max','sensor11_mean','sensor11_min','sensor11_std','sensor12_max','sensor12_mean','sensor12_min','sensor12_std','sensor13_max','sensor13_mean','sensor13_min','sensor13_std','sensor14_max','sensor14_mean','sensor14_min','sensor14_std','sensor15_max','sensor15_mean','sensor15_min','sensor15_std','sensor16_max','sensor16_mean','sensor16_min','sensor16_std','sensor17_max','sensor17_mean','sensor17_min','sensor17_std','sensor18_max','sensor18_mean','sensor18_min','sensor18_std','sensor19_max','sensor19_mean','sensor19_min','sensor19_std','sensor1_max','sensor1_mean','sensor1_min','sensor1_std','sensor20_max','sensor20_mean','sensor20_min','sensor20_std','sensor21_max','sensor21_mean','sensor21_min','sensor21_std','sensor2_max','sensor2_mean','sensor2_min','sensor2_std','sensor3_max','sensor3_mean','sensor3_min','sensor3_std','sensor4_max','sensor4_mean','sensor4_min','sensor4_std','sensor5_max','sensor5_mean','sensor5_min','sensor5_std','sensor6_max','sensor6_mean','sensor6_min','sensor6_std','sensor7_max','sensor7_mean','sensor7_min','sensor7_std','sensor8_max','sensor8_mean','sensor8_min','sensor8_std','sensor9_max','sensor9_mean','sensor9_min','sensor9_std','fold']
    test_df = test_df.iloc[1: , :]
    print(train_df.columns)
    print(train_df.head())   
    test_df.fold.fillna(0, inplace=True)

    # Extracting the label column
    y_test = test_df.pop("broken")

    # convert the dataframe values to array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    #clf = GradientBoostingClassifier(
    #    n_estimators=args.n_estimators, learning_rate=args.learning_rate
    #)
    #clf.fit(X_train, y_train)
    print("Training model")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Saving model with mlflow - leave this section unchanged
    with tempfile.TemporaryDirectory() as td:
        print("Saving model with MLFlow to temporary directory")
        tmp_output_dir = os.path.join(td, "my_model_dir")
        mlflow.sklearn.save_model(sk_model=model, path=tmp_output_dir)

        print("Copying MLFlow model to output path")
        for file_name in os.listdir(tmp_output_dir):
            print("  Copying: ", file_name)
            # As of Python 3.8, copytree will acquire dirs_exist_ok as
            # an option, removing the need for listdir
            shutil.copy2(src=os.path.join(tmp_output_dir, file_name), dst=os.path.join("./outputs", file_name))

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.model, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
```

- Create the training yaml file

```
%%writefile {train_src_dir}/train.yml
# <component>
name: train_nasaturbofan_defaults_model
display_name: Train NASA turbofan Defaults Model
# version: 1 # Not specifying a version will automatically update the version
type: command
inputs:
  train_data: 
    type: uri_folder
  test_data: 
    type: uri_folder
  learning_rate:
    type: number     
  registered_model_name:
    type: string
outputs:
  model:
    type: uri_folder
code: .
environment:
  # for this step, we'll use an AzureML curate environment
  azureml://registries/azureml/environments/AzureML-responsibleai-0.20-ubuntu20.04-py38-cpu/versions/4
  #azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1
  #aml-scikit-learn:0.1.0
command: >-
  python train.py 
  --train_data ${{inputs.train_data}} 
  --test_data ${{inputs.test_data}} 
  --learning_rate ${{inputs.learning_rate}}
  --registered_model_name ${{inputs.registered_model_name}} 
  --model ${{outputs.model}}
```

- Load the component

```
# importing the Component Package
from azure.ai.ml import load_component

# Loading the component from the yml file
train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))
```

- Register the training component

```
# Now we register the component to the workspace
train_component = ml_client.create_or_update(train_component)

# Create (register) the component in your workspace
print(
    f"Component {train_component.name} with Version {train_component.version} is registered"
)
```

- Now create a compute if not exist
- This code is optional

```
from azure.ai.ml.entities import AmlCompute

cpu_compute_target = "cpu-cluster"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        # Name assigned to the compute cluster
        name="cpu-cluster",
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.begin_create_or_update(cpu_cluster)

print(
    f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}"
)
```

- Now create the RAI Pipeline

```
from azure.ai.ml import dsl, Input, Output


@dsl.pipeline(
    compute=cpu_compute_target,
    description="E2E train pipeline",
)
def nasaturbofan_defaults_pipeline(
    pipeline_job_train_data_input,
    pipeline_job_test_data_input,
    pipeline_job_test_train_ratio,
    pipeline_job_learning_rate,
    pipeline_job_registered_model_name,
):
    # using data_prep_function like a python call with its own inputs
    #data_prep_job = data_prep_component(
    #    data=pipeline_job_data_input,
    #    test_train_ratio=pipeline_job_test_train_ratio,
    #)

    # using train_func like a python call with its own inputs
    train_job = train_component(
        train_data=pipeline_job_train_data_input,  # note: using outputs from previous step
        test_data=pipeline_job_test_data_input,  # note: using outputs from previous step
        learning_rate=pipeline_job_learning_rate,  # note: using a pipeline input as parameter
        registered_model_name=pipeline_job_registered_model_name,
    )
```

- Setup the pipeline

```
registered_model_name = "NASAturbofan_defaults_model"

# Let's instantiate the pipeline with the parameters of our choice
pipeline = nasaturbofan_defaults_pipeline(
    pipeline_job_train_data_input=Input(type="mltable", path=my_training_data_input.path),
    pipeline_job_test_data_input=Input(type="mltable", path=my_training_data_test.path),
    pipeline_job_test_train_ratio=0.25,
    pipeline_job_learning_rate=0.05,
    pipeline_job_registered_model_name=registered_model_name,
)
```

- Submit the pipeline

```
import webbrowser

# submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    # Project's name
    experiment_name="e22_NASATurbofan_Training_registered_components",
)
# open the pipeline in web browser
webbrowser.open(pipeline_job.studio_url)
```

- Wait for the pipeline to complete

```
ml_client.jobs.stream(pipeline_job.name)
```

## Responsible AI Dashboard - Model Analysis

- Configure model information

```
expected_model_id = f"NASAturbofan_defaults_model:1"
modelname = "NASAturbofan_defaults_model"
azureml_model_id = f"azureml:{expected_model_id}"
```

- Configure Azure machine learning workspace configuration

```
# Enter details of your AML workspace
subscription_id = "xxxxxxxxxxxxxxxxxxxxxxx"
resource_group = "rgname"
workspace = "azuremlwkspace"
```

- load the registry

```
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

- Let's invoke the Resposible AI Components

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

- Configure score dashbaord information

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

- Setup target and categorical features

```
target_column_name = "broken"
categorical_features = ['sensor14_min', 'sensor13_std', 'sensor15_max', 'sensor6_mean', 'sensor16_std', 'sensor17_max', 'sensor11_std', 'sensor5_min', 'sensor9_min', 'sensor10_min', 'sensor3_mean', 'sensor17_std', 'sensor4_min', 'sensor11_mean', 'sensor6_max', 'sensor3_min', 'sensor12_min', 'sensor14_mean', 'sensor3_std', 'sensor10_mean', 'sensor2_max', 'sensor11_max', 'sensor7_mean', 'unit_number', 'sensor18_std', 'sensor10_max', 'sensor2_min', 'sensor2_mean', 'sensor12_max', 'sensor13_mean', 'sensor19_mean', 'sensor5_max', 'sensor13_max', 'sensor1_mean', 'sensor7_max', 'sensor18_min', 'sensor12_mean', 'sensor11_min', 'sensor15_std', 'sensor8_mean', 'sensor12_std', 'sensor21_max', 'sensor2_std', 'sensor13_min', 'sensor8_std', 'sensor4_max', 'sensor20_min', 'sensor19_max', 'sensor18_mean', 'sensor19_min', 'sensor15_min', 'sensor21_std', 'sensor10_std', 'sensor17_min', 'sensor5_std', 'sensor1_std', 'sensor16_mean', 'sensor4_mean', 'sensor8_min', 'sensor7_min', 'sensor4_std', 'sensor17_mean', 'sensor18_max', 'sensor5_mean', 'sensor15_mean', 'sensor14_std', 'sensor19_std', 'sensor20_std', 'sensor3_max', 'sensor21_min', 'sensor21_mean', 'sensor9_max', 'sensor9_mean', 'sensor16_max', 'sensor1_max', 'sensor6_min', 'sensor8_max', 'lifetime', 'sensor20_max', 'sensor20_mean', 'fold', 'sensor6_std', 'sensor1_min', 'sensor7_std', 'sensor9_std', 'sensor16_min', 'sensor14_max']
```

- create a mode suffix

```
import time

model_name_suffix = int(time.time())
```

- create a pipeline

```
import json
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import dsl, Input

classes_in_target = json.dumps(["Less than median", "More than median"])
treatment_features = json.dumps(
    ["sensor14_min", "sensor11_mean", "sensor11_max", "sensor11_min", "sensor11_std"]
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

- Invoke the pipeline

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

# Workaround to enable the download
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

- submit the pipeline

```
nasaturbofanrai_job = ml_client.jobs.create_or_update(insights_pipeline_job)

print(f"Created job: {nasaturbofanrai_job}")
```

- wait for the job to complete

```
ml_client.jobs.stream(nasaturbofanrai_job.name)
```

