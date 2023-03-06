# Azure Open AI Dall-E with Power Apps

## Let's build a Power App to use Azure Open AI Dall-E for various use cases

## What's needed

- Register for Open AI - https://beta.openai.com/
- At the time of writing this article dalle is private preview
- Create a deployment inside the resource

## Create a Power App

- To create a power app first need to create a power flow
- Flow is invoked by a powerapp trigger
- Text information will be passed to the flow

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as dalleapi
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle1.jpg "Architecture")

- Now create a variable to hold the text passed from power app into flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle3.jpg "Architecture")

- Now send the text to Dall E Api using Http action

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle4.jpg "Architecture")

- here is the api we are using

```
https://api.openai.com/v1/images/generations
```

- Need headers for Authentication and Content-Type

```
Bearer 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
Content-Type: application/json
```

- Body information

```
{
  "prompt": @{variables('prompt')},
  "n": 2,
  "size": "1024x1024"
}
```

- now bring parseJSOn action to parse the response

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle5.jpg "Architecture")

- here is the schema

```
{
    "type": "object",
    "properties": {
        "created": {
            "type": "integer"
        },
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string"
                    }
                },
                "required": [
                    "url"
                ]
            }
        }
    }
}
```

- Initialize a variable to hold the url

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle6.jpg "Architecture")

- Now do a foreach for each image and get the last one

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle7.jpg "Architecture")

- now parse the output url variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle8.jpg "Architecture")

- here is the schema to parse json

```
{
    "type": "object",
    "properties": {
        "url": {
            "type": "string"
        }
    }
}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle9.jpg "Architecture")

## Power App

- Now let's create a power app
- Create a text box to enter the text: Man walking in moon
- then create a button to invoke the flow - name is create

```
Set(imageurl,dalleapi.Run(TextInput2.Text))
```

- Now create an image control to display the image

```
Concatenate("<img src='", imageurl.url , "' style='width:400px;height:400px;'></img>")
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle10.jpg "Architecture")

- Final output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/dalle2.jpg "Architecture")