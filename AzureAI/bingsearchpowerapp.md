# Power App to use bing api and Summarize results with Azure Open AI GPT 3

## Let's build a Power App to use Azure Open AI ChatGPT to summarize the results from Pine Cone index

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as South Central US
- At the time of writing this article gpt4, gpt3.5-turbo is only available in south central US
- Create a deployment inside the resource
- Create bing resource
- Get the API uri and key

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as bingsearch
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch3.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch4.jpg "Architecture")

- Now lets initialize a variable called searchtxt

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch5.jpg "Architecture")

- Now lets call the bing search api

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch6.jpg "Architecture")

- Set the URL as below

```
https://api.bing.microsoft.com/v7.0/search?count=5&q=@{variables('searchtxt')}
```

- We are using GEt method for search with returing 5 results
- Set the header as below

```
Content-Type: multipart/form-data
Ocp-Apim-Subscription-Key: <your_api_key>
```

- Now add ParseJSON to parse the response
- Here is the schema to parse. Schema can change

```
{
    "type": "object",
    "properties": {
        "_type": {
            "type": "string"
        },
        "queryContext": {
            "type": "object",
            "properties": {
                "originalQuery": {
                    "type": "string"
                }
            }
        },
        "webPages": {
            "type": "object",
            "properties": {
                "webSearchUrl": {
                    "type": "string"
                },
                "totalEstimatedMatches": {
                    "type": "integer"
                },
                "value": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string"
                            },
                            "name": {
                                "type": "string"
                            },
                            "url": {
                                "type": "string"
                            },
                            "isFamilyFriendly": {
                                "type": "boolean"
                            },
                            "displayUrl": {
                                "type": "string"
                            },
                            "snippet": {
                                "type": "string"
                            },
                            "deepLinks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string"
                                        },
                                        "url": {
                                            "type": "string"
                                        },
                                        "snippet": {
                                            "type": "string"
                                        },
                                        "deepLinks": {
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "name",
                                        "url",
                                        "snippet"
                                    ]
                                }
                            },
                            "dateLastCrawled": {
                                "type": "string"
                            },
                            "language": {
                                "type": "string"
                            },
                            "isNavigational": {
                                "type": "boolean"
                            }
                        },
                        "required": [
                            "id",
                            "name",
                            "url",
                            "isFamilyFriendly",
                            "displayUrl",
                            "snippet",
                            "dateLastCrawled",
                            "language",
                            "isNavigational"
                        ]
                    }
                }
            }
        },
        "relatedSearches": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string"
                },
                "value": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string"
                            },
                            "displayText": {
                                "type": "string"
                            },
                            "webSearchUrl": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "text",
                            "displayText",
                            "webSearchUrl"
                        ]
                    }
                }
            }
        },
        "rankingResponse": {
            "type": "object",
            "properties": {
                "mainline": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answerType": {
                                        "type": "string"
                                    },
                                    "resultIndex": {
                                        "type": "integer"
                                    },
                                    "value": {
                                        "type": "object",
                                        "properties": {
                                            "id": {
                                                "type": "string"
                                            }
                                        }
                                    }
                                },
                                "required": [
                                    "answerType",
                                    "value"
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch7.jpg "Architecture")

- Now initialize a variable called searchout
- Set the value as below

```
body('Parse_JSON')?['webPages']?['value']?[0]?['snippet']
```

- Now for each the value array and pull all the snippets

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch13.jpg "Architecture")

- Value to select
- Loop through the value array
- Select snippet from the value array
- Append to string searchouttxt

- i am only picking one row of result. If need more please loop deep links objects and grab the information


![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch8.jpg "Architecture")

- Now send it to Azure Open AI api to summarize the text
- Here is the api
- Use Post

```
https://aoairesourcename.openai.azure.com/openai/deployments/deploymentname/completions?api-version=2022-12-01
```

- Set the header as below

```
Content-Type: application/json
api-key: <your_api_key>
```

- now body as below

```
{
  "prompt": @{concat('Summarize ' , variables('searchouttxt'))},
  "max_tokens": 500,
  "temperature": 0
}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch9.jpg "Architecture")

- Now parse the results

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch10.jpg "Architecture")

- here is the schema used. please validate the schema as it can change

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

- Now initialize a variable called outsummary
- Then bring Apply for Each and assign the text output to output variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch11.jpg "Architecture")

- Finally send the outsummary to power apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch12.jpg "Architecture")

## Power Apps

- Create a new app
- Create a input text box and a button
- Create a label to display the output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/bingsearch1.jpg "Architecture")

- Now create a flow and connect to the flow
- Assign the flow to button

```
Set(messagevar,bingsearch.Run(TextInput1_2.Text,TextInput1_2.Text));
```

- Now set the text box to display the output
- set the text property of the text box as below

```
messagevar.outputsearch
```

- Goal is to use bing search to get current information get the top 5 docs and then pass that get summarized by azure open ai. Then display the output in power apps