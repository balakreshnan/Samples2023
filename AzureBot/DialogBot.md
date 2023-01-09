# Azure AIOPS Guided Assistant

## Create a Bot for AIOPS Guided Assistant

## Introduction

- This is a guided assistant for AIOPS operation web site
- Using Waterfall Dialogs to guide the user through a multi-step process
- Easy option to select and go is provided instead NLP
- This is for begining level of bot development
- NLP will be added later stage

## Pre-requisites

- Azure Account
- Install Azure Bot Composer in Development computer
- Source - https://learn.microsoft.com/en-us/composer/install-composer?tabs=windows
- Make sure net 3.1 core and Node js both are installed for Composer to work
- We can publish the bot to Azure Bot Service
- Also download and install Bot Framework Emulator
- Create a LUIS or CLU Azure cognitive services
- Use that in bot composer

## Create a Bot

- Open Azure Bot Composer
- Name the Bot as AIOPS Bot
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot1.jpg "Output Episodes")

- Above image shows bunch of diaglogs to create and use in the bot
- We will see how to create those later.
- By default, it will create a root dialog
- It will also have the greetings trigger
- First we have to create a new dialog called TopLevelServices
- Here is how the Greeting dialog looks like

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot2.jpg "Output Episodes")

- Now let's see the Top Level Services Dialog

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot3.jpg "Output Episodes")

- As you can see above, we have a trigger called TopLevelServices
- Let's add Prompt to ask user to select the service
- Will be a simple multi option prompt
- Then bring the condition option for switch case
- For prompt make sure you set the property and also select index as value

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot5.jpg "Output Episodes")

- Then for the multi option prompt, we have to add the options

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot4.jpg "Output Episodes")

- We will using the index value in switch case to navigate to next level dialog
- Make sure you create new dialog for dataservice, networkservice and customerservice
- Let the dialog be empty for now
- In Switch (multi option) case, we have to add the cases as 0,1,2
- Make sure the Condition is set to user.choice is that we assigned in previous prompt

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot6.jpg "Output Episodes")

- Next for every case value we will add corresponding dialogs.
- For the case 0 we will add a dialog called DataServices

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot7.jpg "Output Episodes")

- For the case 1 we will add a dialog called NetworkService

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot8.jpg "Output Episodes")

- For the case 2 we will add a dialog called CustomerService

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot9.jpg "Output Episodes")

- Above are all dialog which we will see their implementation later
- Go to Dataservice dialog and add a prompt to ask user to select the service

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot10.jpg "Output Episodes")

- Now add multi option prompt to ask user to select the service
- Most of these dialogs can be converted to NLP based in later case
- Let's create options
- Add a property to refernce in switch case
  
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot11.jpg "Output Episodes")

- Now add the options to use

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot12.jpg "Output Episodes")

- Next add multi option swtich case
- select the options as user.choice1
- values are usually index values
- For each index value we will add a dialog

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot13.jpg "Output Episodes")

- Repeat the same for NetworkService and CustomerService
- For NetworkService, we will add a prompt to ask user to select the service
- For now these are empty dialogs
- For CustomerService, we will add a prompt to ask user to select the service
- For now these are empty dialogs
- Now we are going to have another level in Data services and we are only going to expand Compliance option
- Add a prompt to ask user to select the service

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot14.jpg "Output Episodes")

- add property name to store the index values
- These values will be used in switch case to select the next dialog

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot15.jpg "Output Episodes")

- now add the options to choose from

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot16.jpg "Output Episodes")

- Then add switch multiple option case
- for index 0 add onpremise as dialog
- Make sure you create a dialog called onpremise
- you can only select from the list of dialogs created

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot17.jpg "Output Episodes")

- for onpremise dialog we are going to do some data processing
- We are going to access a HTTP REST API
- Process the value and display
- we are going to get current time and create dynamics text
- So create a muti option prompt to ask user to select the service

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot18.jpg "Output Episodes")

- Set the property with new name

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot19.jpg "Output Episodes")

- Add the options

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot20.jpg "Output Episodes")

- Now bring the multi choice swith case
- Add the index values
- For each index we are going to do multiple data process
- First bring http request action

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot21.jpg "Output Episodes")

- here i am using a free API to just get some data
- Method is GET
- URL

```
https://api.agify.io/?name=kiran
```

- now process the result and send response back
- Add send response
- Just diplay which option was selected with current utc time

```
${concat('You are in Compliance ', utcNow())}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot22.jpg "Output Episodes")

- Then we are going to process the REST API response
- Send the response back to user

```
${concat(' Process JSON property Name: ', turn.results.content.name, ' now Age: ' ,turn.results.content.age)}
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureBot/images/aiopsbot23.jpg "Output Episodes")

- Next to figure out how to go back to previous dialog