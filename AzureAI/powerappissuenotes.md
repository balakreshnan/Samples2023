# Azure Open AI ChatGPT with Power Apps

## Let's build a Power App to use Azure Open AI ChatGPT for various use cases

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as East US
- At the time of writing this article davinci-003 is only available in East US
- Create a deployment inside the resource

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as getsummary
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion1.jpg "Architecture")

- First add trigger as Power Apps
- then Initialize a variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion2.jpg "Architecture")

- for value assign from Power apps
- that will take the input value and assign to the variable called prompt
- Now lets send the data to openai API to use davinci model using GPT-3
- First bring HTTP action
- Then select the action as POST
- here is the URL 

```
https://resourcename.openai.azure.com/openai/deployments/davinci003/completions?api-version=2022-12-01
```

- Note we need content-type:application/json
- also need api-key: <your_api_key>
- here is the body

```
{
  "prompt": @{triggerBody()['Initializevariable_Value']},
  "max_tokens": 100,
  "temperature": 1
}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion3.jpg "Architecture")

- make sure the prompt property is substituted with the value of the variable prompt as shown above
- Next we need to parse the response from above HTTP output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion4.jpg "Architecture")

- Now we need to provide a sample document to parse the JSON schema

```
{
  "id": "cmpl-xxxxxxxxxxx",
  "object": "text_completion",
  "created": 1640707195,
  "model": "davinci:2020-05-03",
  "choices": [
    {
      "text": " really bright. You can see it in the sky at night.\nJupiter is the third brightest thing in the sky, after the Moon and Venus.\n",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ]
}
```

- Schema generated from sample

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

- initalize a variable called outsummary

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion5.jpg "Architecture")

- select the Type as String
- After parsing we need to loop the array and assign the text to the variable
- Bring Apply to each action
- Select Choices as the array property
- now bring Set variable action
- Assign the currentitem to the variable outsummary

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion6.jpg "Architecture")

- Now add Parse JSON action

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion7.jpg "Architecture")

- Schema to parse

```
{
    "type": "object",
    "properties": {
        "text": {
            "type": "string"
        },
        "index": {
            "type": "integer"
        },
        "finish_reason": {
            "type": "string"
        },
        "logprobs": {}
    }
}
```

- Next add Respond to Power Apps
- Sent the outsumamry as response back to Power Apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion8.jpg "Architecture")

- Save the flow
- Do a manual test run by passing sample text
- If successful then you are set with flow

## Power Apps

- Now lets create a Power App
- This is only a simple app
- i am creating a canvas app
- Name the app as: OpenAITest
- Create a text box to input text
- then create on the right 4 buttons and 4 label to store output
- For each button Issue, Solution, Benefit, and Proof

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/issues1.jpg "Architecture")

- For Issues

```
Set(summarytext,Openaisummarization.Run(Concatenate("Extract Issue: ", TextInput10.Text)));UpdateContext({cleantxt: summarytext.summarytext});
```

- For Solution

```
Set(summarytext1,Openaisummarization.Run(Concatenate("Extract Solution: ", TextInput10.Text)));UpdateContext({cleantxt: summarytext1.summarytext});
```

- For Benefit

```
Set(summarytext2,Openaisummarization.Run(Concatenate("Extract Benefits: ", TextInput10.Text)));UpdateContext({cleantxt: summarytext2.summarytext});
```

- For Proof

```
Set(summarytext3,Openaisummarization.Run(Concatenate("Extract Proof: ", TextInput10.Text)));UpdateContext({cleantxt: summarytext3.summarytext});
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/issues3.jpg "Architecture")