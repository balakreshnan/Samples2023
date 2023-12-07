# Azure Databricks gpt 3.5 fine tuning

## Introduction

- Using Azure data bricks, fine tune gpt 3.5 model
- Using Open AI SDK version 0.28.1
- Using Northcentral US region
- Sample data set
- Code from: https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune?tabs=python%2Ccommand-line


## Requirements

- Azure Subscription
- Azure Databricks workspace
- Azure open ai resource
- training and validation jsonl files
- open ai version 0.28.1
- Current version doesn't work
- Upload the jsonl file into azure databricks

## Code

- install necessay library

```
%pip install "openai==0.28.1" json requests os tiktoken time
```

- restart the kernel

```
dbutils.library.restartPython()
```

- Validate the open ai version

```
pip show openai
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/ADB/Images/finetune1.jpg "Architecture")

- Loading the dataset

```
import json

# Load the training set
with open('/Workspace/Users/babal@microsoft.com/training_set.jsonl', 'r', encoding='utf-8') as f:
    training_dataset = [json.loads(line) for line in f]

# Training dataset stats
print("Number of examples in training set:", len(training_dataset))
print("First example in training set:")
for message in training_dataset[0]["messages"]:
    print(message)

# Load the validation set
with open('/Workspace/Users/babal@microsoft.com/validation_set.jsonl', 'r', encoding='utf-8') as f:
    validation_dataset = [json.loads(line) for line in f]

# Validation dataset stats
print("\nNumber of examples in validation set:", len(validation_dataset))
print("First example in validation set:")
for message in validation_dataset[0]["messages"]:
    print(message)
```

- Utilities to calculate tokens used

```
import json
import tiktoken
import numpy as np
from collections import defaultdict

encoding = tiktoken.get_encoding("cl100k_base") # default encoding used by gpt-4, turbo, and text-embedding-ada-002 models

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

files = ['/Workspace/Users/babal@microsoft.com/training_set.jsonl', '/Workspace/Users/babal@microsoft.com/validation_set.jsonl']

for file in files:
    print(f"Processing file: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    total_tokens = []
    assistant_tokens = []

    for ex in dataset:
        messages = ex.get("messages", {})
        total_tokens.append(num_tokens_from_messages(messages))
        assistant_tokens.append(num_assistant_tokens_from_messages(messages))
    
    print_distribution(total_tokens, "total tokens")
    print_distribution(assistant_tokens, "assistant tokens")
    print('*' * 50)
```

- Configure the open ai

```
import openai 

openai.api_type = "azure"
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
openai.api_base = "https://aoairesourcename.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
```

- Setting up the fine tuning

```
# Upload fine-tuning files
import openai
import os

#openai.api_key = os.getenv("AZURE_OPENAI_API_KEY") 
#openai.api_base =  os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-09-15-preview' # This API version or later is required to access fine-tuning for turbo/babbage-002/davinci-002

training_file_name = '/Workspace/Users/babal@microsoft.com/training_set.jsonl'
validation_file_name = '/Workspace/Users/babal@microsoft.com/validation_set.jsonl'

# Upload the training and validation dataset files to Azure OpenAI with the SDK.

training_response = openai.File.create(
    file=open(training_file_name, "rb"), purpose="fine-tune", user_provided_filename="training_set.jsonl"
)
training_file_id = training_response["id"]

validation_response = openai.File.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune", user_provided_filename="validation_set.jsonl"
)
validation_file_id = validation_response["id"]

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)
```

- Create the fine tuning job

``` 
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-35-turbo-0613",
)

job_id = response["id"]

# You can use the job ID to monitor the status of the fine-tuning job.
# The fine-tuning job will take some time to start and complete.

print("Job ID:", response["id"])
print("Status:", response["status"])
print(response)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/ADB/Images/finetune2.jpg "Architecture")

- Monitor the fine tuning job

```
response = openai.FineTuningJob.retrieve(job_id)

print("Job ID:", response["id"])
print("Status:", response["status"])
print(response)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/ADB/Images/finetune3.jpg "Architecture")

- Track the fine tuning job

```
# Track training status

from IPython.display import clear_output
import time

start_time = time.time()

# Get the status of our fine-tuning job.
response = openai.FineTuningJob.retrieve(job_id)

status = response["status"]

# If the job isn't done yet, poll it every 10 seconds.
while status not in ["succeeded", "failed"]:
    time.sleep(10)
    
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)
    print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
    status = response["status"]
    print(f'Status: {status}')
    clear_output(wait=True)

print(f'Fine-tuning job {job_id} finished with status: {status}')

# List all fine-tuning jobs for this resource.
print('Checking other fine-tune jobs for this resource.')
response = openai.FineTuningJob.list()
print(f'Found {len(response["data"])} fine-tune jobs.')
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/ADB/Images/finetune4.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/ADB/Images/finetune5.jpg "Architecture")

- Output of the fine tuning job

```
#Retrieve fine_tuned_model name

response = openai.FineTuningJob.retrieve(job_id)

print(response)
fine_tuned_model = response["fine_tuned_model"]
```

- Output

```
{
  "hyperparameters": {
    "n_epochs": 2
  },
  "status": "succeeded",
  "model": "gpt-35-turbo-0613",
  "fine_tuned_model": "gpt-35-turbo-0613.ft-xxxxxxxxxxxxxxxxxxxx",
  "training_file": "file-xxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "validation_file": "file-xxxxxxxxxxxxxxxxxxxxxxxx",
  "result_files": [
    "file-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ],
  "finished_at": 1701960883,
  "trained_tokens": 1336,
  "id": "ftjob-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "created_at": 1701958237,
  "updated_at": 1701960883,
  "object": "fine_tuning.job"
}
```