# Power App Search Cognitive Search using Vector Seach and Azure Open AI Embeddings

## Let's build a Power App to use Azure Cognitive Search with Vector Search using Azure Open AI Embeddings

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as East US
- At the time of writing this article davinci-003 is only available in East US
- Create a deployment inside the resource
- Create Cognitive Search
- Create a Index to work with

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as vectorsearch
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear3.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear4.jpg "Architecture")

- Now we need to add a trigger as Power Apps
- Initialize a variable called searchtxt

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear5.jpg "Architecture")

- Now call HTTP action to get the embeddings from Azure Open AI

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear6.jpg "Architecture")

- Set the URL as below

```
https://aoairesourcename.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01
```

- next Headers

```
content-type:application/json
api-key: <your_api_key>
```

- Now set the body as

```
{
  "input": @{variables('searchtxt')},
  "model": "text-embedding-ada-002"
}
```

- Now we need to parse the response from above HTTP output
- To keep the embedding simple we will use the first embedding from the response
- Let's not parse rather use the other action called "Parse JSON (single value)"
- Parse Json

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear7.jpg "Architecture")

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
                    "embedding": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },
                    "index": {
                        "type": "integer"
                    }
                },
                "required": [
                    "object",
                    "embedding",
                    "index"
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

- Now we need to get the first embedding from the response
- Pass that to Next HTTP call to cognitive search
- Here is the configuration

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear8.jpg "Architecture")

- Configure the HTTP

```
https://cogsearchresourcename.search.windows.net/indexes/vectorindexname/docs/search?api-version=2023-07-01-Preview
```

- Setup headers

```
content-type:application/json
api-key: <your_api_key>
```

- Set the body as

```
{
  "vector": {
    "value": @{body('Parse_JSON')?['data'][0]['embedding']},
    "fields": "contentVector",
    "k": 10
  },
  "select": "title, category, content"
}
```

- Now we need to parse the response from above HTTP output
- Bring ParseJSON action

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear9.jpg "Architecture")

- here is the schema

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
                    "title": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string"
                    }
                },
                "required": [
                    "@@search.score",
                    "title",
                    "content",
                    "category"
                ]
            }
        }
    }
}
```

- Initialize a outputt variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear10.jpg "Architecture")

- Now we need to loop through the response and add the results to the output variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear11.jpg "Architecture")

- Now we need to return the output variable
- Return the output to power app

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear12.jpg "Architecture")

## Power App

- Create a power app
- Add a input text box

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsearc1.jpg "Architecture")

- Bring a button to process

```
Set(messagevar,vectorsearch.Run(TextInput14.Text));
```

- Add a label for output
- Set the label to below output

```
messagevar.searchoutput
```
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/vectorsear1.jpg "Architecture")