# Azure Machine learning Running LLamaCode 7b and 34b

## Introduction

- Fine tune LLama2 Code model in Azure ML
- Using Azure ML
- Using NVdia A100 GPU
- SKU Standard_NC48ads_A100_v4 (48 cores, 440 GB RAM, 128 GB disk)
- I had to request quota increase using Azure ML to achieve this experiment
- using open source data set
- Following this experiment from [here](https://huggingface.co/codellama/CodeLlama-34b-hf)
- Idea here is show how we can run this in Azure Machine learning compute instance

## Code

- first install necesary packages

```
%pip install git+https://github.com/huggingface/transformers.git@main accelerate
```

- restart the kernel
- i choose python 3
- then run the following code

```
from transformers import AutoTokenizer
import transformers
import torch
```

- now bring the model

```
#model = "codellama/CodeLlama-7b-hf"
model = "codellama/CodeLlama-34b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureML/Images/llama2code1.jpg "Architecture")

- now time to invoke the code
- pass a simple code and see the output

```
sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

- here is the output

```
def ping_exponential_backoff(host: str):
    """
    Ping a host using exponential backoff.

    :param host: The host to ping.
    :return: True if the host is reachable, False otherwise.
    """
    timeout = 1
    while True:
        try:
            socket.create_connection((host, 80), timeout=timeout)
            return True
        except OSError:
            timeout *= 2
            if timeout > 1024:
                return False


def ping_host(host: str):
    """
    Ping a host.

    :param host: The host to ping.
    :return: True if the host is reachable, False otherwise.
    """
    try:
        socket.create_connection((host, 80), timeout
```

- Switch the model name and i tested 7billion and 34 billion model
- make sure there is enough space to download the model