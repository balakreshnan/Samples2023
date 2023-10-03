# Azure Open AI Whisper Model in Azure Machine Learning

## Azure Machine learning consuming azureopen ai whisper model

### introduction

- Using Azure open ai whisper model in Azure ML
- Using Azure Machine Learning notebook
- Using Pythin Code
- Using Azure Open AI Whisper model
- Deployment name: whisper
- Model name: whisper

## Code

- Upgrade your open ai sdk to latest version

```
%pip install --upgrade openai
```

- import Azure open ai configuration

```
from dotenv import dotenv_values
# specify the name of the .env file name 
env_name = "env2.env" # change to use your own .env file
config = dotenv_values(env_name)
```

- i am using whisper in East US region
- might change based on where you have it deployed and available

- Now import open ai library and set the configuration

```
import openai

openai.api_type = "azure"
openai.api_key = config["AZURE_OPENAI_API_KEY"]
openai.api_base = config["AZURE_OPENAI_ENDPOINT"]
openai.api_version = "2023-09-01-preview"
```

- Also load the environment variables

```
import os

os.environ["OPENAI_API_BASE"] = config["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = config["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = "2023-09-01-preview"
os.environ["OPENAI_API_TYPE"] = "azure"
```

- Now create a transcribe function

```
def transcribe(audio):
    with open(audio, "rb") as audio_file:
        transcription = openai.Audio.transcribe(
            file=audio_file,
            deployment_id="whisper",
            model="whisper"
        )
    # print(transcription["text"])
    return transcription["text"]
```

- now we pass the audio file to the transcribe function
- i have a local folder called data and i have a file called wikipediaOcelot.wav

```
rs = transcribe('data/wikipediaOcelot.wav')
```

- print the output text

```
print(rs)
```