# Azure Power App using Florence model

## Let's build a Power App to use Azure Vision Cognitive service Florence Model for various use cases

## What's needed

- Create a Azure Cognitive Service resource for vision in Azure portal
- Get the URI and keys to use

## Create a Power App

- To create a power app first need to create a power flow
- Flow is invoked by a powerapp trigger
- Images is passed to the flow
- on the default screen assign values to variables
- Add an Add media with image control
- Text box to display the output
- Button to invoke the flow
  
```
Set(JSONImageSample, Substitute(JSON(UploadedImage1.Image, JSONFormat.IncludeBinaryData), """", ""));
Set(outputtext,florencemodel4.Run(JSONImageSample));
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence1.jpg "Architecture")

## Power Flow

- Flow name is florencemodel4
- Let go deep into the flow
- Note we are passing image as binary URI in json format
- we need to make sure the proper image format is sent to florence model api
- First create a variable and assign the output from power app
- Variable name is: imagedata

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence3.jpg "Architecture")

- now add Compose action
- This is a important step to make sure the image is sent in proper format

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence4.jpg "Architecture")

- Function to convert dataurito bring

```
dataUriToBinary(variables('imagedata'))
```

- Now bring the HTTP connector to connect to florence api

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence5.jpg "Architecture")

- For URL: https://modelresname.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2022-10-12-preview&features=objects&language=en
- Content-Type: application/octet-stream
- Ocp-Apim-Subscription-Key: xxxxxxxx
- Body: @{body('Compose_2')}
- For response we need to parse the response
- Bring ParseJson

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence6.jpg "Architecture")

- here is the schema

```
{
    "type": "object",
    "properties": {
        "modelVersion": {
            "type": "string"
        },
        "metadata": {
            "type": "object",
            "properties": {
                "width": {
                    "type": "integer"
                },
                "height": {
                    "type": "integer"
                }
            }
        },
        "objectsResult": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "confidence": {
                                "type": "number"
                            },
                            "boundingBox": {
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "integer"
                                    },
                                    "y": {
                                        "type": "integer"
                                    },
                                    "w": {
                                        "type": "integer"
                                    },
                                    "h": {
                                        "type": "integer"
                                    }
                                }
                            }
                        },
                        "required": [
                            "name",
                            "confidence",
                            "boundingBox"
                        ]
                    }
                }
            }
        }
    }
}
```

- Now set the variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence7.jpg "Architecture")

- Now apply for Each
- now compose the output with mulitple output
- then assign the variable to output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence8.jpg "Architecture")

```
{
  "Name": @{items('Apply_to_each_2')?['name']},
  "probability": @{items('Apply_to_each_2')?['confidence']},
  "boundingbox": @{items('Apply_to_each_2')?['boundingBox']}
}
```

```
outputs('Compose')
```

- now send the output back to power apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence9.jpg "Architecture")

## Power Apps continuation

- Now we have the output from flow
- apply that to the text box

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/florence1.jpg "Architecture")

- Keep trying with new images and see the objects.
- In future we will see how to use this for other use cases