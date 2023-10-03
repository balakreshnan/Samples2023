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

- output

```
The ocelot, Lepardus paradalis, is a small wild cat native to the southwestern United States, Mexico, and Central and South America. This medium-sized cat is characterized by solid black spots and streaks on its coat, round ears, and white neck and undersides. It weighs between 8 and 15.5 kilograms, 18 and 34 pounds, and reaches 40 to 50 centimeters – 16 to 20 inches – at the shoulders. It was first described by Carl Linnaeus in 1758. Two subspecies are recognized, L. p. paradalis and L. p. mitis. Typically active during twilight and at night, the ocelot tends to be solitary and territorial. It is efficient at climbing, leaping, and swimming. It preys on small terrestrial mammals such as armadillo, opossum, and lagomorphs.
```