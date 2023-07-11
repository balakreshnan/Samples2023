# Deploy Open AI Whisper V2 Manage Endpoint

## Steps

- Download whisper V2 large model
- Create a conda.yaml file for the environment
- create a score file for the endpoint
- Deploy it as managed endpoint
- Sample audio file to test the endpoint

## Score file

```
import os
import logging
import json
import numpy
from typing import  Dict
from transformers.pipelines.audio_utils import ffmpeg_read
import whisper
import torch
import shutil
import base64

SAMPLE_RATE = 16000

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # deserialize the model file back into a sklearn model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'vca_whisper.pkl')

    model = whisper.load_model("large-v2", download_root=os.getenv("AZUREML_MODEL_DIR"))
    logging.info("Init complete")


def run(data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    #logging.info(data)
    inputs = base64.b64decode(data)
    audio_nparray = ffmpeg_read(inputs, SAMPLE_RATE)
    audio_tensor= torch.from_numpy(audio_nparray)
    
    # run inference pipeline
    result = model.transcribe(audio_nparray)

    # postprocess the prediction
    return {"text": result["text"]}
```

## Conda.yaml

```
channels:
- anaconda
- pytorch
- conda-forge
dependencies:
- python=3.8.16
- pip<=23.0.1
- ffmpeg=4.2.2
- pip:
  - mlflow==2.3.1
  - cloudpickle==2.2.1
  - jsonpickle==3.0.1
  - mlflow-skinny==2.3.1
  - azureml-core==1.51.0.post1
  - azureml-mlflow==1.51.0
  - azureml-metrics==0.0.14.post1
  - scikit-learn==0.24.2
  - cryptography==41.0.1
  - python-dateutil==2.8.2
  - datasets==2.11.0
  - soundfile==0.12.1
  - librosa==0.10.0.post2
  - diffusers==0.14.0
  - sentencepiece==0.1.97
  - transformers==4.30.2
  - torch==2.0.1
  - torchaudio
  - accelerate==0.20.3
  - Pillow==9.4.0
  - azureml-evaluate-mlflow==0.0.14.post1
  - wget==3.2
  - more-itertools==9.1.0
  - ffmpeg-python==0.2.0
  - azureml-inference-server-http
  - openai-whisper
name: mlflow-env
```

## Deployment Code

- Write code to deploy to Managed endpoint
- We are using CPU
- install openai-whisper
  
  ```
  pip install openai-whisper
  or 
  pip install git+https://github.com/openai/whisper.git 
  ```
- import libraries

```
# import required libraries
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment, CodeConfiguration, OnlineRequestSettings, ProbeSettings
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import datetime
```

- Create a client

```
subscription_id = "xxxxxxxxxxxxxx"
resource_group = "rgname"
workspace = "workspacename"

# get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
```

- Set credentials

```
credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")
registry_ml_client = MLClient(credential, registry_name="azureml")
```

- download openai-whisper model

```
import whisper 

modelwhisper = whisper.load_model("large-v2")

import pickle

pickle.dump(modelwhisper,open('vca_whisper.pkl','wb'))
```

- Create a model

```
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

file_model = Model(
    path="vca_whisper.pkl",
    type=AssetTypes.CUSTOM_MODEL,
    name="vca_whisper",
    description="Model created from Open AI Whispher for speech to text",
)
ml_client.models.create_or_update(file_model)
```

- Create endpoint name

```
# Define an endpoint name
endpoint_name = "whisper-largecli" + datetime.datetime.now().strftime("%m%d%H%M%f")
```

- Create Endpoint

```
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="An online endpoint for custom deployment of scoring script for batch inference for whisper",
    auth_mode="key",
)
```

```
ml_client.begin_create_or_update(endpoint)
```

- get model to deploy

```
model_name = "vca_whisper"
custommodel = ml_client.models.get(name=model_name, version="1") 
print(custommodel.id)
```

- Create environment

```
environment = Environment(
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest",
)
```

- Create a deployment

```
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=custommodel.id,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=90000,
        max_concurrent_requests_per_instance=1,
        max_queue_wait_ms=500,
    ),
    environment=environment,
    instance_type="Standard_DS5_v2",
    code_configuration=CodeConfiguration(
       code="./",
       scoring_script="onlineScore.py"
    ),
    instance_count=1,
)
```

```
ml_client.online_deployments.begin_create_or_update(blue_deployment)
```

- wait for deployment to complete
- then assign traffic

```
endpoint.traffic = { "blue": 100 }

ml_client.begin_create_or_update(endpoint).result()
```

## now test endpoint

- Endpoint configuration

```
endpoint = "https://endpointname.eastus2.inference.ml.azure.com/score"
key= "xxxxx"
```

- Test endpoint

```
import urllib.request
import requests as r

import json
import os
import ssl
import base64
import mimetypes

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

url = 'https://endpointname.eastus2.inference.ml.azure.com/score'
#api_key = '{{key}}' # Replace this with the API key for the web service
api_key = key

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules

audiofile = './sample1.flac'
with open(audiofile, "rb") as i:
      b = i.read()

# get mimetype
content_type= mimetypes.guess_type(audiofile)[0]
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue' }

#response = r.get(url, headers=headers, data=b)
body= base64.b64encode(b)
req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

- Delete the resources

```
# delete the endpoint and the deployment
ml_client.online_endpoints.begin_delete(endpoint_name)
```