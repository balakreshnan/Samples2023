# Semantic Kernel in Azure Machine Learning Notebook and adding memory with huggingface models

## Use Case

- Show case how to consume semantic kernel in Azure Machine Learning Notebook
- Using python sdk
- Use huggingface models
- Add memory to the model

## Code

- First install semantic kernel

```
%pip install --upgrade semantic-kernel
```

```
%pip install torch==2.0.0
```

```
%pip install transformers==4.28.1
```

```
%pip install sentence-transformers==2.2.2
```

```
%pip install accelerate
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

- import libraries

```
import semantic_kernel as sk
import semantic_kernel.connectors.ai.hugging_face as sk_hf
```

- configure LLM for memory

```
# Configure LLM service
kernel.add_text_completion_service(
    "gpt2", sk_hf.HuggingFaceTextCompletion("gpt2", task="text-generation")
)
kernel.add_text_embedding_generation_service(
    "sentence-transformers/all-MiniLM-L6-v2",
    sk_hf.HuggingFaceTextEmbedding("sentence-transformers/all-MiniLM-L6-v2"),
)
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())
```

- Save data into memory

```
await kernel.memory.save_information_async(
    "animal-facts", id="info1", text="Sharks are fish."
)
await kernel.memory.save_information_async(
    "animal-facts", id="info2", text="Whales are mammals."
)
await kernel.memory.save_information_async(
    "animal-facts", id="info3", text="Penguins are birds."
)
await kernel.memory.save_information_async(
    "animal-facts", id="info4", text="Dolphins are mammals."
)
await kernel.memory.save_information_async(
    "animal-facts", id="info5", text="Flies are insects."
)

# Define semantic function using SK prompt template language
my_prompt = """I know these animal facts: {{recall $query1}} {{recall $query2}} {{recall $query3}} and """

# Create the semantic function
my_function = kernel.create_semantic_function(
    my_prompt, max_tokens=45, temperature=0.5, top_p=0.5
)
```

- Run the model and ask a question

```
context = kernel.create_new_context()
context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "animal-facts"
context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 0.3

context["query1"] = "animal that swims"
context["query2"] = "animal that flies"
context["query3"] = "penguins are?"
output = await kernel.run_async(my_function, input_vars=context.variables)

output = str(output).strip()


query_result1 = await kernel.memory.search_async(
    "animal-facts", context["query1"], limit=1, min_relevance_score=0.3
)
query_result2 = await kernel.memory.search_async(
    "animal-facts", context["query2"], limit=1, min_relevance_score=0.3
)
query_result3 = await kernel.memory.search_async(
    "animal-facts", context["query3"], limit=1, min_relevance_score=0.3
)

print(f"gpt2 completed prompt with: '{output}'")
```