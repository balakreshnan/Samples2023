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

```
Note: this process can be applied to any HTTP REST enabled actions needed to be invoked by Power Apps
```

- Now we need to create a canvas
- Bring Text Input Box
- Add default text as prompt
- Set the default to a variable called: openaitext

```
openaitext
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion9.jpg "Architecture")

- Now we are going to Test
- Summarize Text
- Create SQL
- Classify Text
- Parse Unstructed data
- Classify content

- Now we add a button for Summarize Text
- Call the flow and assign the return value to the variable
- Here is the value to set

```
UpdateContext({openaitext: "A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.

Tl;dr"})
```

- Now add another button to create SQL from normal lanugage text

```
UpdateContext({openaitext: "### Postgres SQL tables, with their properties:
#
# Employee(id, name, department_id)
# Department(id, name, address)
# Salary_Payments(id, employee_id, amount, date)
#
### A query to list the names of the departments which employed more than 10 employees in the last 3 months

SELECT"})
```

- Now add another button to Classify Text

```
UpdateContext({openaitext:"Classify the following news article into 1 of the following categories: categories: [Business, Tech, Politics, Sport, Entertainment]

news article: Donna Steffensen Is Cooking Up a New Kind of Perfection. The Internet’s most beloved cooking guru has a buzzy new book and a fresh new perspective:

Classified category:"})
```

- Now add another button to Parse Unstructed Data

```
UpdateContext({openaitext:"There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them.

Please make a table summarizing the fruits from Goocrux
| Fruit | Color | Flavor |
| Neoskizzles | Purple | Sweet |
| Loheckles | Grayish blue | Tart |"})
```

- Now add another button to classify content

```
UpdateContext({openaitext: "The following is a list of companies and the categories they fall into

Facebook: Social media, Technology
LinkedIn: Social media, Technology, Enterprise, Careers
Uber: Transportation, Technology, Marketplace
Unilever: Conglomerate, Consumer Goods
Mcdonalds: Food, Fast Food, Logistics, Restaurants
FedEx:"})
```


- Add a Submit button
- Openaisummarization is the name of the flow

```
Set(summarytext,Openaisummarization.Run(TextInput1.Text))
```

- in OnSelect apply the above formulat.
- Openaisummarization is the name of the flow and we are passing parameters as TextInput1.text
- Now lets add a Text lable as label1
- Assign the text property to summarytext.summarytext
- summarytext is the output property set in the flow

```
summarytext.summarytext
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion10.jpg "Architecture")

- Save the canvas app
- Run the app and test it
- below should be the output

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/openaicompletion11.jpg "Architecture")

- The above flow can be used to access most API's in open AI.
- So does we can use this for other Cognitive services