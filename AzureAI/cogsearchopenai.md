# Power App Search Cognitive Search and Summarize results

## Let's build a Power App to use Azure Open AI ChatGPT to summarize the results from Cognitive Search

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
- Name it as cogsearchsummary
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch3.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch4.jpg "Architecture")

- Now we need to add a trigger as Power Apps
- Now initialize a variable caleed searchtxt

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch5.jpg "Architecture")

- Call the Cognitive Search API to get the results

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch6.jpg "Architecture")

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
  "search": @{variables('searchtxt')},
  "skip": 0,
  "top": 5
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
                    "content": {
                        "type": "string"
                    },
                    "metadata_storage_path": {
                        "type": "string"
                    },
                    "people": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "organizations": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "keyphrases": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "masked_text": {
                        "type": "string"
                    },
                    "merged_content": {
                        "type": "string"
                    },
                    "text": {
                        "type": "array"
                    },
                    "layoutText": {
                        "type": "array"
                    },
                    "pii_entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string"
                                },
                                "type": {
                                    "type": "string"
                                },
                                "subtype": {},
                                "offset": {
                                    "type": "integer"
                                },
                                "length": {
                                    "type": "integer"
                                },
                                "score": {
                                    "type": "number"
                                }
                            },
                            "required": [
                                "text",
                                "type",
                                "subtype",
                                "offset",
                                "length",
                                "score"
                            ]
                        }
                    }
                },
                "required": [
                    "@@search.score",
                    "content",
                    "metadata_storage_path",
                    "people",
                    "organizations",
                    "locations",
                    "keyphrases",
                    "masked_text",
                    "merged_content",
                    "text",
                    "layoutText",
                    "pii_entities"
                ]
            }
        }
    }
}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch7.jpg "Architecture")

- Assign HTTP body as input
- Now set a variable called searchouttxt as below

```
searchouttxt
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch8.jpg "Architecture")

- Now parse the output from searchouttxt
- Use Apply for Each and assign the output from searchouttxt as input

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch16.jpg "Architecture")

- now join the text for summaization

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch10.jpg "Architecture")

- command

```
substring(variables('searchouttxt'),0,3000)
```

- Now we need to call the Open AI API to summarize the text

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch11.jpg "Architecture")

- Set the URL as below

```
https://aoiresourcename.openai.azure.com/openai/deployments/davinci003/completions?api-version=2022-12-01
```

- Set the header as below

```
content-type:application/json
api-key: <your_api_key>
```

- Set Body as

```
{
  "prompt": @{concat('Summarize ' , variables('searchouttxt'))},
  "max_tokens": 500,
  "temperature": 0
}
```

- Now parse the jSON output from above HTTP action

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch12.jpg "Architecture")

- Here is the JSON schema

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

- Now set a variable called outsummary as below
- Now loop the output and get the summary


![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch13.jpg "Architecture")

- Now send the output to power apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch14.jpg "Architecture")

## Power Apps

- Create a new app
- Create a input text box and a button
- Create a label to display the output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch1.jpg "Architecture")

- Now create a flow and connect to the flow
- Assign the flow to button

```
Set(messagevar,cogsearch.Run(TextInput1_2.Text,TextInput1_2.Text));
```

- Now set the label to display the output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/cogsearch2.jpg "Architecture")

- Goal is to bring top 5 results and summarize the text