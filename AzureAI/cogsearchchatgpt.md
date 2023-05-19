# Power App Search Cognitive Search and Summarize results with ChatGPT 3.5 turbp/Gpt4

## Let's build a Power App to use Azure Open AI ChatGPT to summarize the results from Cognitive Search

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as East US
- At the time of writing this article gpt4, gpt3.5-turbo is only available in south central US
- Create a deployment inside the resource
- Create Cognitive Search
- Create a Index to work with
- Enable Semantic Search as well

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as cogsearchsummarychatgpt
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea1.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea2.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea3.jpg "Architecture")

- Now we need to add a trigger as Power Apps
- Now initialize a variable caleed searchtxt

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea4.jpg "Architecture")

- Now bring parse JSON

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea5.jpg "Architecture")

- here is the schema

```
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string"
            },
            "role": {
                "type": "string"
            }
        },
        "required": [
            "content",
            "role"
        ]
    }
}
```

- Now parse the output and save to output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea6.jpg "Architecture")

- now use the output to get search keyword

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea8.jpg "Architecture")

- Send to chatgpt to get keywork
- Use HTTP action

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea9.jpg "Architecture")

- use post
- Here is the url

```
https://aoairesorucename.openai.azure.com/openai/deployments/deploymentname/chat/completions?api-version=2023-03-15-preview
```

- Set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- now body

```
{
  "messages": @{outputs('Compose_2')},
  "max_tokens": 1000,
  "temperature": 0.7
}
```

- Now parse the output and create search keyword

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea10.jpg "Architecture")

- Here is the schema

```
{
    "type": "object",
    "properties": {
        "id": {
            "type": "string"
        },
        "object": {
            "type": "string"
        },
        "created": {
            "type": "integer"
        },
        "model": {
            "type": "string"
        },
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer"
                    },
                    "finish_reason": {
                        "type": "string"
                    },
                    "message": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string"
                            },
                            "content": {
                                "type": "string"
                            }
                        }
                    }
                },
                "required": [
                    "index",
                    "finish_reason",
                    "message"
                ]
            }
        },
        "usage": {
            "type": "object",
            "properties": {
                "completion_tokens": {
                    "type": "integer"
                },
                "prompt_tokens": {
                    "type": "integer"
                },
                "total_tokens": {
                    "type": "integer"
                }
            }
        }
    }
}
```

- Now parse the information and save to variable to send to search

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea11.jpg "Architecture")

- Call the Cognitive Search API to get the results

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsea12.jpg "Architecture")

- Set the URL as below

```
https://searchsvc.search.windows.net/indexes/indexname/docs/search?api-version=2020-06-30
```

- Set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- Set Body as

```
{
  "search": @{variables('searchtxt2')},
  "top": 3,
  "queryType": "semantic",
  "semanticConfiguration": "default",
  "queryLanguage": "en-us",
  "speller": "lexicon",
  "captions": "extractive|highlight-false"
}
```

- Now we need to parse the response from above HTTP output
- bring parse JSON action

```
{
    "type": "object",
    "properties": {
        "@@odata.context": {
            "type": "string"
        },
        "value": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "@@search.score": {
                        "type": "number"
                    },
                    "id": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
                    },
                    "sourcefile": {
                        "type": "string"
                    }
                },
                "required": [
                    "@@search.score",
                    "id",
                    "content",
                    "sourcefile"
                ]
            }
        }
    }
}
```

- Some time schema can change based on your index, take a sample and use that to create the schema

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt5.jpg "Architecture")

- Now time to loop the results and create single string
- Initialize a variable called searchouttxt
- Bring Apply for Each
- Select Content or value
- Add Append to string variable
- Select Content

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt6.jpg "Architecture")

- initialize a variable 4 is not needed
- Now let's bring compose to get the message format setup

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt7.jpg "Architecture")

- now the prompt engineered message
- Also strip double quotes, new line and carriage return

```
[
  {
    "role": "system",
    "content": "You are an assistant that helps company employees answer questions and draft marketing/sales messages in response to RFP documents. You must use the provided sources below to answer the question. Return the response as a bulleted list of paragaphs including citations for each supporting facts. If there isn't enough information in the sources below, respond that you aren't sure and give your best answer. You must always cite your sources. Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response.   Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]. For tabular information return it as an html table. Do not return markdown format."
  },
  {
    "role": "user",
    "content": "@{uriComponentToString(replace(uriComponent(replace(variables('searchouttxt'), '"', '')), '%0A', ''))}"
  }
]
```

- Now it's time to call the chatgpt api

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt8.jpg "Architecture")

- Set the URL as below

```
https://aoairesourcename.openai.azure.com/openai/deployments/deploymentname/chat/completions?api-version=2023-03-15-preview
```

- Set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- Set Body as

```
{
  "messages": @{outputs('Compose')},
  "max_tokens": 1000,
  "temperature": 0
}
```

- lets get the output from the chatgpt and pass it back into power apps
- Iinitialize a variable called outsummary
- assign the output from chatgpt to this variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt9.jpg "Architecture")

- Next send it back to power apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt10.jpg "Architecture")

## Power Apps

- Create a new app
- Create a input text box and a button
- Create a label to display the output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogserchatgpt11.jpg "Architecture")

- Now create a flow and connect to the flow
- Assign the flow to button

```
Clear(convlist1);Collect(convlist1, { role : "user", content: TextInput1.Text } );Set(messagevar, cogsearchsummarychatgpt.Run(JSON(convlist1)));Set(outvar2, Text(ParseJSON(messagevar.output).choices.'0'.message.content));Collect(convlist1, { role : "assistant", content: outvar2});
```

- Now set the text box to display the output
- set the text property of the text box as below

```
outvar2
```

- Goal is to use semantic search get the top 3 docs and then pass that with prompt engineering to chatgpt and get the response back and display it in power apps