# Semantic Kernel in Azure Machine Learning Notebook

## Use Case

- Show case how to consume semantic kernel in Azure Machine Learning Notebook
- Using python sdk

## Code

- First install semantic kernel

```
%pip install --upgrade semantic-kernel
```

- Restart the kernel
- Load Environment variable

```
from dotenv import dotenv_values
# specify the name of the .env file name 
env_name = "env.env" # change to use your own .env file
config = dotenv_values(env_name)
```

- assign environment variables

```
import os

# Set the ENV variables that Langchain needs to connect to Azure OpenAI
os.environ["OPENAI_API_BASE"] = config["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = config["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = config["AZURE_OPENAI_API_VERSION"]
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = config["AZURE_OPENAI_DEPLOYMENT_NAME"]
```

- now setup Azure open ai configuration

```
OPENAI_API_KEY=config["AZURE_OPENAI_API_KEY"]
OPENAI_ORG_ID=""
AZURE_OPENAI_DEPLOYMENT_NAME=config["AZURE_OPENAI_DEPLOYMENT_NAME"]
AZURE_OPENAI_ENDPOINT=config["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY=config["AZURE_OPENAI_API_KEY"]
```

- import the libraries

```
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
```

- set the kernel configuration for Azure Open AI

```
kernel = sk.Kernel()

kernel.add_chat_service(                      # We are adding a text service
    "gpt-35-turbo-16k",                            # The alias we can use in prompt templates' config.json
    AzureChatCompletion(
        AZURE_OPENAI_DEPLOYMENT_NAME,                 # Azure OpenAI *Deployment name*
        AZURE_OPENAI_ENDPOINT,  # Azure OpenAI *Endpoint*
        AZURE_OPENAI_API_KEY         # Azure OpenAI *Key*
    )
)
```

- set the chat model

```
kernel.set_default_text_completion_service("gpt-35-turbo-16k")
```

- now create a prompt

```
# Wrap your prompt in a function
prompt = kernel.create_semantic_function("""
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words.""")

# Run your prompt
print(prompt()) # => Robots must not harm humans.
```

- now send that kernel

```
# Create a reusable function with one input parameter
summarize = kernel.create_semantic_function("{{$input}}\n\nOne line TLDR with the fewest words.")

# Summarize the laws of thermodynamics
print(summarize("""
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""))

# Summarize the laws of motion
print(summarize("""
1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
2. The acceleration of an object depends on the mass of the object and the amount of force applied.
3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first."""))

# Summarize the law of universal gravitation
print(summarize("""
Every point mass attracts every single other point mass by a force acting along the line intersecting both points.
The force is proportional to the product of the two masses and inversely proportional to the square of the distance between them."""))

# Output:
# > Energy conserved, entropy increases, zero entropy at 0K.
# > Objects move in response to forces.
# > Gravitational force between two point masses is inversely proportional to the square of the distance between them.
```

- output

```
Energy conserved, entropy increases, zero entropy at absolute zero.
Newton's laws of motion: objects stay still or keep moving unless acted upon, acceleration depends on mass and force, and forces come in pairs.
Masses attract each other based on distance.
```