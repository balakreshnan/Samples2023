# Azure Open AI with Power Apps

## Let's build a Power App to use Azure Open AI for various use cases

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as East US
- At the time of writing this article davinci-003 is only available in East US
- Create a deployment inside the resource

## Create a Power App

- To create a power app first need to create a power flow
- Flow is invoked by a powerapp trigger
- Text information will be passed to the flow
- on the default screen assign values to variables

```
Set(inputvar, "{ \""messages\"": [{ \""role\"": \""system\"", \""content\"": \""You are a helpful assistant.\"" }, { \""role\"": \""user\"", \""content \"": \""hi there\"" } ]}"); Collect(convlist, { role : "system", content: " You are a helpful assistant." }); Set(outvar, "");
```

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as chatgptprocessing
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp2.jpg "Architecture")

- First add trigger as Power Apps
- then Initialize a variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp3.jpg "Architecture")

- Bring Parse JSON to parse the input variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp4.jpg "Architecture")

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

- Now lets send the data to openai API to use davinci model using chatgpt model

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp5.jpg "Architecture")

- First bring HTTP action
- Then select the action as POST
- here is the URL 

```
https://resourcename.openai.azure.com/openai/deployments/chatgpt/chat/completions?api-version=2023-03-15-preview
```

- Note we need content-type:application/json
- also need api-key: <your_api_key>
- here is the body

```
{
  "messages": @{body('Parse_JSON')}
}
```

- Now initialize a variable to store the response

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp6.jpg "Architecture")

- name the variable as output
- set the value as ""

- now assign the variable to the output variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp7.jpg "Architecture")

- Now assign the put to respond to power app

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp8.jpg "Architecture")

- Now lets test the flow
- Save the flow
- Move to power apps

## Power Apps

- NOw in the power app drag a label and assign the value to the variable outvar
- Now in the default screen assign values to variables

```
Set(inputvar, "{ \""messages\"": [{ \""role\"": \""system\"", \""content\"": \""You are a helpful assistant.\"" }, { \""role\"": \""user\"", \""content \"": \""hi there\"" } ]}"); Collect(convlist, { role : "system", content: " You are a helpful assistant." }); Set(outvar, "");
```

- Bring Text box and assign the value to inputvar
- Create a button to call the flow

```
Collect(convlist, { role : "user", content: TextInput9.Text } );Set(outputvar, chatgptprocessing.Run(JSON(convlist)));Set(outvar, Text(ParseJSON(outputvar.output).choices.'0'.message.content));Collect(convlist, { role : "assistant", content: outvar});
```

- Now create a gallery to show the conversation
- drop the vertical gallery
- assing the value to convlist

```
convlist
```

- Now bring a create a clear button

```
Clear(convlist);Collect(convlist, { role : "system", content: " I am Chat bot - helpful assistant." });Set(outvar, "");
```

- Now save and test the app

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp1.jpg "Architecture")