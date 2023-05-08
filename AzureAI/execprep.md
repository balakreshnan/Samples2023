# Azure Open AI ChatGPT with Power Apps

## Let's build a Power App to use Azure Open AI ChatGPT for various use cases

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as South Central US
- At the time of writing this article davinci-003 is only available in East US/SouthCentral US/West europe
- Create a deployment inside the resource call it davinci003 for Text-davinici-003

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as Openaisummarization
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
- Sent the outsumamry as response back to Power Apps with variables called: summarytext

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion8.jpg "Architecture")

- Save the flow
- Do a manual test run by passing sample text
- If successful then you are set with flow

## Power Apps

- Create a new Power App Screen
- i am creating a canvas app
- Name the app as: Exec Prep
- Add a text input control for Company name input
- Add a button and name it "Go"
- in the onclick set the variable called Openaisummarization to the output of the flow

```
Set(summarytext,Openaisummarization.Run(Concatenate("Firstname:Lastname:Title
Michael:Jordan:ceo
Michael:Smith:VP
MIchael:M2:SVP
List all ",TextInput6.Text," executives in separate line in above format")));UpdateContext({execlist: summarytext.summarytext});
```

- Create a another list text box
- assign the output from above the variable called execlist
- Textbox default property is

```
execlist
```

- Now create another Text input control for Executive name
- Create another button called "Profile"
- Now we are going to grab the profile of the Executive
- Set the onclick of the profile button

```
Set(summarytext,Openaisummarization.Run(Concatenate("Create a profile of ",TextInput8.Text)));UpdateContext({profiletxt: summarytext.summarytext});
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/execprep1.jpg "Architecture")

- Create a label text and assign the previous variable called profiletxt

```
profiletxt
```

- Now add another button called "Get Current Priorities"
- in on click of the button

```
Set(summarytext,Openaisummarization.Run(Concatenate("what are priorities of ", TextInput8.Text , " for ",TextInput6.Text)));UpdateContext({pritxt: summarytext.summarytext});
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/execprep2.jpg "Architecture")

- Create a label text and assign the previous variable called pritxt, on default property

```
pritxt
```

- Create a button and call it "Get Quarterly Results"

```
Set(summarytext,Openaisummarization.Run(Concatenate("Extract Insights from quarterly earnings for ",TextInput6.Text)));UpdateContext({pritxt: summarytext.summarytext});
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/execprep3.jpg "Architecture")

- Create a button and call it "10K Insights"

```
Set(summarytext,Openaisummarization.Run(Concatenate("Extract insights from 10K report for ",TextInput6.Text)));UpdateContext({pritxt: summarytext.summarytext});
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/execprep4.jpg "Architecture")

- Create a button and call it "Recommendations"
- on the on click

```
Set(summarytext,Openaisummarization.Run(Concatenate("match Accenture recommendations for the below priorities: ",pritxt)));UpdateContext({pritxt: summarytext.summarytext});
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/execprep5.jpg "Architecture")