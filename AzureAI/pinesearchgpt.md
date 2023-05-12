# Power App Pine Cone and Summarize results with Azure Open AI GPT 3

## Let's build a Power App to use Azure Open AI ChatGPT to summarize the results from Pine Cone index

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as East US
- At the time of writing this article gpt4, gpt3.5-turbo is only available in south central US
- Create a deployment inside the resource
- Create pine cone service
- Create a Index to work with
- Enable Cosine Similarity as well

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as pineconegpt
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt1.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt2.jpg "Architecture")

- Now we need to add a trigger as Power Apps
- Now initialize a variable caleed searchtxt

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt3.jpg "Architecture")

- Call the embedding service to get the vector

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt4.jpg "Architecture")

- Set the URL as below

```
https://aoaoresourcename.openai.azure.com/openai/deployments/text-deploy/embeddings?api-version=2023-03-15-preview
```

- Set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- Set Body as

```
{
  "input": @{variables('searchtxt')}
}
```

- now parse the vecotr from the response

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt5.jpg "Architecture")

- here is the schema

```
{
    "type": "object",
    "properties": {
        "object": {
            "type": "string"
        },
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string"
                    },
                    "index": {
                        "type": "integer"
                    },
                    "embedding": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    }
                },
                "required": [
                    "object",
                    "index",
                    "embedding"
                ]
            }
        },
        "model": {
            "type": "string"
        },
        "usage": {
            "type": "object",
            "properties": {
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

- now we need to call the pinecone service to get the results
- send the vector as input

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt6.jpg "Architecture")

- set the URL as below

```
https://indexname.svc.us-east1-gcp.pinecone.io/query
```

- set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- body

```
{
  "includeMetadata": true,
  "includeValues": true,
  "namespace": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "topK": 5,
  "vector": @{body('Parse_JSON_3')?['data']?[0]?['embedding']}
}
```

- now parse the json response

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt7.jpg "Architecture")

- set the schema as

```
{
    "type": "object",
    "properties": {
        "results": {
            "type": "array"
        },
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string"
                    },
                    "score": {
                        "type": "number"
                    },
                    "values": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string"
                            },
                            "text": {
                                "type": "string"
                            }
                        }
                    }
                },
                "required": [
                    "id",
                    "score",
                    "values",
                    "metadata"
                ]
            }
        },
        "namespace": {
            "type": "string"
        }
    }
}
```

- Now initialize a varibale
- Apply for each
- Take each text append the text to variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt9.jpg "Architecture")

- now send to gpt 2 summarization service

- set the URL as below

```
https://aoairesourcename.openai.azure.com/openai/deployments/deployment/completions?api-version=2022-12-01
```

- set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- body

```
{
  "prompt": @{concat('Summarize the content in 300 words: ' , variables('searchouttxt'))},
  "max_tokens": 300,
  "temperature": 0
}
```

- now parse the json output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt10.jpg "Architecture")

- schema for parsing

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
                    "text": {
                        "type": "string"
                    },
                    "index": {
                        "type": "integer"
                    },
                    "logprobs": {},
                    "finish_reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "text",
                    "index",
                    "logprobs",
                    "finish_reason"
                ]
            }
        }
    }
}
```

- now
- initialize a variable calle outsummary
- apply for each
- append to variable: outsummary

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt11.jpg "Architecture")

- now send the output back to powerapp
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt12.jpg "Architecture")

## Power Apps

- Create a new app
- Create a input text box and a button
- Create a label to display the output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/pinegpt13.jpg "Architecture")

- Now create a flow and connect to the flow
- Assign the flow to button

```
Set(messagevar1,pineconegpt.Run(TextInput1_1.Text));
```

- Now set the text box to display the output
- set the text property of the text box as below

```
Substitute(Substitute(TrimEnds(Trim(messagevar1.outputsearch)), Char(10),""), """","")
```

- Goal is to use semantic search get the top 3 docs and then pass that with prompt engineering to chatgpt and get the response back and display it in power apps