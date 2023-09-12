# Create RFP responses with your own data and create word document using Azure OpenAI

## Introduction

- Create a UI in power apps to capture the RFP data
- Ability to load RFP and create responses
- Ability to search company data
- Ask question to extract information from your own data
- Create a word document with the RFP data
- Refine and improve the questions or update the text and regenerate the document
- Data used are sample profiles i took from publicly available linkedin profiles.

## Prerequisites

- Azure Account
- Power Apps Account
- Power Flow HTTP connector
- Azure Open AI service - GPT 4 Model 
- Azure Cognitive Search - Semantic Search

## Power App UI

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-1.jpg "Architecture")

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as aecbidgpt41
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-6.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-7.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-8.jpg "Architecture")

- Sequence of the flow is as follows
- Take user question -> Get semantic search keyword text -> Get the answer from the RFP data (Cognitive Search) -> Send the asnwers to Azure open ai to summarize

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-9.jpg "Architecture")

- now get the user question and assign to a variable
- format the message to send to open ai
- Here is the parseJSON schema

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

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-10.jpg "Architecture")

- Now lets create the chatgpt message

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-11.jpg "Architecture")

- Now create the message as follows

```
[
  {
    "role": "system",
    "content": "You are an assistant that helps X Industry Bot"
  },
  {
    "role": "user",
    "content": @{concat('Your job is to generate a short keyword search query based on the question or comment from the user. Only return the suggested search query with no other content or commentary. Do not answer the question or comment on it. Just generate the keywords for a search query. User question or comment: ', variables('searchtxt1'), ' Search query:')}
  }
]
```

- Let's call Azure open ai to get the semantic search keyword

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-12.jpg "Architecture")

- Method is POST
- URI is https://aoainame.openai.azure.com/openai/deployments/depname/chat/completions?api-version=2023-07-01-preview
- Set Content-Type to application/json
- Set api-key to the key from Azure open ai
- Set Body

```
{
  "messages": @{outputs('Create_Message_to_send_to_Open_AI')},
  "max_tokens": 1000,
  "temperature": 0.7
}
```

- Parse the JSON output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-13.jpg "Architecture")

- Schema

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

- Now parse the output and save the text to a variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-14.jpg "Architecture")

- now send the semantic keyword to Azure cognitive search to get the answer
- we are using semantic search feature

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-15.jpg "Architecture")

- parse the output from search

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-16.jpg "Architecture")

- Schema used:

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

- Parse the output and save the text to a variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-17.jpg "Architecture")

- Create an array to send to Azure open ai to summarize

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-18.jpg "Architecture")

- Here is the body to frame for chatgpt message

```
[
  {
    "role": "system",
    "content": "You are an assistant that helps X employees answer questions and draft marketing/sales messages in response to RFP documents. You must use the provided sources below to answer the question. Return the response as a bulleted list of paragaphs including citations for each supporting facts. If there isn't enough information in the sources below, respond that you aren't sure and give your best answer. You must always cite your sources. Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response.   Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]. For tabular information return it as an html table. Do not return markdown format."
  },
  {
    "role": "user",
    "content": "@{uriComponentToString(replace(uriComponent(replace(variables('searchouttxt'), '"', '')), '%0A', ''))}"
  }
]
```

- Now call Azure open ai to summarize the text

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-19.jpg "Architecture")

- now set the method as post
- uri is https://aoainame.openai.azure.com/openai/deployments/deployname/chat/completions?api-version=2023-07-01-preview
- Set the content-type to application/json
- Set the api-key to the key from Azure open ai
- Form the body as below:

```
{
  "messages": @{outputs('ComposeMessageforChatGPT')},
  "max_tokens": 1000,
  "temperature": 0.7
}
```

- Parse the output
- Get the summarized text
- Send the summarized text to power apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-20.jpg "Architecture")

- Now go to power apps

## Power Apps

- Create a new app
- add the flow created to the power app
- Here is the screen to build

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-1.jpg "Architecture")

- Create 3 button in the Top for sample questions
- Each text box:

```
Set(searchtxt, "List top 5 candidates for leadership?")
```

```
Set(searchtxt, "List top 5 candidates for Strategy?")
```

```
Set(searchtxt, "List top 5 candidates for Technology leader?")
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-21.jpg "Architecture")

- Then create a text box and search button
- for the text box assign the variable

```
searchtxt
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-22.jpg "Architecture")

- Then add a another multiline text box to display the answer

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-2.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-3.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-4.jpg "Architecture")

- Now final screen to consolidate and create word document

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-5.jpg "Architecture")

- Grab the output and create a word document
- Or save the information to temp variable and then create a word document automatically
- Create a button to invoke word document creation

```
Set(summarytext6, PopulateWordrfp.Run(TextInput10.Text))
```

- make sure all the text needed are saved in TextInput10.Text
- Flow details for PopulateWordrfp

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-23.jpg "Architecture")

- Here is the flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-24.jpg "Architecture")

- i am saving word template in onedrive
- template has section as elements to be filled by content sent from power apps
- Every new instance create a word document
- check and see the word document created

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpapp1-25.jpg "Architecture")