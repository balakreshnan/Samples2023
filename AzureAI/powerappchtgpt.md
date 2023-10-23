# Azure Open AI ChatGPT with Power Apps

## Let's build a Power App to use Azure Open AI ChatGPT for various use cases

## What's needed

- Register for Azure Open AI - https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview
- Once got approved create a azure open ai resource in Azure portal
- Select region as East US
- At the time of writing this article davinci-003 is only available in East US
- Create a deployment inside the resource

## Create a Power App

- First create a blank canvas app at https://make.preview.powerapps.com/
- Then create the a power flow **inside of the canvas app(important!!)**
- This flow will be invoked by a power app trigger, so that text input from users will be passed to the flow. 


## Create a Power Flow inside of the Power App

- Let's create a power flow **inside of the Power App**
- On the left menu in power apps click on Power Automate
- Click on Create New Flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp9.jpg "Architecture")

- Name it as chatgptprocessing
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp2.jpg "Architecture")

- First add trigger as Power Apps
- then Initialize a variable
- For the "Value" field of the variable, click "Ask in PowerApps" to autogenerate a variable named "initializevariable_Value".  

![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp10.jpg "Architecture")

- After the auto-generation, your screen should look like this:

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

## Create the UI in the Power App

- In your Power App interface, under the "Tree view", select "App" (instead of "Screen1"), go to "Advanced" tab on the right, under "OnStart", put the following code. Doing this will initial values to the variables `inputvar` and `outvar` when the app is started.   
```
Set(inputvar, "{ \""messages\"": [{ \""role\"": \""system\"", \""content\"": \""You are a helpful assistant.\"" }, { \""role\"": \""user\"", \""content \"": \""hi there\"" } ]}"); 
Collect(convlist, { role : "system", content: " I am a helpful assistant." }); 
Set(outvar, "");
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp11.jpg "Architecture")

- Now in the power app click Insert to drag a "Text label" onto your canvas, and assign the value to the variable `outvar`

![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp12.jpg "Architecture")

- Bring Text box and assign the value to inputvar

![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp13.jpg "Architecture")

- Create a Send button to call the flow

```
Collect(convlist, { role : "user", content: TextInput1.Text } );
Set(outputvar, chatgptprocessing.Run(JSON(convlist)));
Set(outvar, Text(ParseJSON(outputvar.output).choices.'0'.message.content));
Collect(convlist, { role : "assistant", content: outvar});
```

- Now create a gallery to show the conversation by clicking Insert to drop a vertical gallery
- assing the value to `convlist`
- Adjust the positioning of the gallery components as you'd like 

![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp14.jpg "Architecture")

- Now bring a create a clear button

```
Clear(convlist);Collect(convlist, { role : "system", content: " I am Chat bot - helpful assistant." });Set(outvar, "");
```

- Now save and test the app

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/chatpgp1.jpg "Architecture")


## Additional features

### To allow users to see previous messages in the conversation history

- Select the arrow in the vertical gallery
- Go to Advanced tab on the right
- Under "OnSelect", put the following code:

```
Set(outvar, ThisItem.content)
```

- Doing this will allow users to click on any of the arrows in the conversation list, and display that message on the right. 
![Architecture](https://github.com/balakreshnan/Samples2023/blob/vxg-edits/AzureAI/images/chatpgp15.jpg "Architecture")