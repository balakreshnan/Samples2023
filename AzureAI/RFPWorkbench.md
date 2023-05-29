# Create RFP with your own data and create word document using Azure OpenAI

## Introduction

- Create a UI in power apps to capture the RFP data
- Ability to search company data
- Ask question to extract information from your own data
- Create a word document with the RFP data
- Refine and improve the questions or update the text and regenerate the document

## Prerequisites

- Azure Account
- Power Apps Account
- Power Flow HTTP connector
  

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as rfpchatgptprocessing
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
- Move to creating next power app for building word document

## Flow to create word document and populate template elements with data from power apps

- Here is the entire flow
- Take a peak into Apply to Each

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc2.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc3.jpg "Architecture")

- Exploded view of Apply to Each

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc4.jpg "Architecture")

- initialize a variable to store the response from power apps called : galleryselected

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc5.jpg "Architecture")

- This variable will have a JSON of all selected items from gallery

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc6.jpg "Architecture")

- Schema for the ParseJSON

```
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Value": {
                "type": "string"
            }
        },
        "required": [
            "Value"
        ]
    }
}
```

- Now initialize a variable for Task, Context and Summary variables
- Each of these variables will be populated from response selected to send to word template

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc7.jpg "Architecture")

- now lets process all the rows and pick the corresponding data from the response
- Save it to variables above
- here is the high level filter to only take assistant response ( we don't need user response)

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc8.jpg "Architecture")

- now lets only take Task data and assign to variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc9.jpg "Architecture")

- Next for Context variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc10.jpg "Architecture")

- Next for Summary variable

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc11.jpg "Architecture")

- now we need to create a word template
- Create a word document
- Enable developer settings
- Click on design mode in developer tab
- click plain text and add a name
- add 2 more text to make it 3 template element
- Save the file as docx
- upload to a location in onedrive, where you have access to

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc12.jpg "Architecture")

- Select the one drive option
- Select the location where it is stored.
- if your document is in proper template format, you will see the template elements
- Now assign each template element the corresponding variable
- Now let's create a file which is word document

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc13.jpg "Architecture")

- finally send to power apps

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc14.jpg "Architecture")

## Power Apps

- Now lets create a power app
- we need one textbox for input search
- another one for output
- One Gallery to store the converstation
- One button to send the input to flow
- now we need to set the page with your own data
- here is the initial colelctions

```
Clear(convlist1);Collect(convlist1, { role : "system", content: "Answer the question based on the context below. Keep the answer short and concise. Provide followup questions: Tableau is an integrated business intelligence (BI) and analytics solution that helps to analyze key business data and generate meaningful insights. The solution helps businesses to collect data from multiple source points such as SQL databases, spreadsheets, cloud apps like Google Analytics and Salesforce to create a collective dataset. Tableau's live visual analytics and interactive dashboard allow slicing & dicing datasets for generating relevant insights and exploring new opportunities. Users can create interactive maps and analyze data across regions, territories, demographics and more. Tableau helps to create a narrative story of the data analysis with interactive visualizations that can be shared with their audience. Tableau can be customized to serve a
tableau Overall Rating: 4.55.
tableau Ease-of-use Rating: 4
tableau Customer Support Rating: 4
tableau Value for money Rating: 4
tableau Functionality Rating: 4.5
tableau
Review 1: I have worked on domains like Telecom and Network where in I have been using Tableau. Its really liked by senior management as it shows each and every bit very clearly to relevant stake holders. For example, Marketing head likes region wise revenue sale dashboard which gives region/sales manager/product wise sale.
PROS:There are many features which lot of users would have liked:- 1. Connection to almost all database 2. Many types of Charts can be created including Sankey, Doughnut, Maps etc. 3. Its has the feature of data blending  and data extraction 4. Dashboards can be viewed on multiple devices like tablet, mobile and laptops. 5. Its really fast while drilling down or filtering out from any dashboard.
CONS:Its a bit expensive tool as compared to other data visualization tools however its a value for money.Reason for choosing TableauThere were multiple check points before choosing Tableau:- 1. We refereed Gartner quadrant which showcased good image of tableau. 2. Reviews taken from friends who are working in other organisation. 3. Some POCs developed on Trail version of TableauReasons for switching to TableauIt was a organisation demand however but as and when I started working on Tableau, I really liked working on it.
Review 2: I believe Tableau is an excellent data visualization tool that is easy to use and provides powerful features for creating interactive visualizations and dashboards quickly.
PROS:Tableau is an outstanding data visualization tool that makes data analysis simple and straightforward. It offers an intuitive drag-and-drop user interface and powerful workbook features to let users develop interactive visualizations and dashboards quickly. I particularly admire Tableau's ability to create appealing visualizations effortlessly. It lets users generate an array of charts, graphs, and maps that allow them to deliver the data's story in no time. Tableau also gives users access to a wide range of analytical abilities like forecasting, trends, and correlations. It is also convenient to share visualizations with the public, enabling users to build interactive web-based visualizations with ease. To top it off, Tableau has a formidable suite of tools for data preparation which allows users to promptly clean, structure, and format data for analysis.
CONS:One of the least liked features of Tableau is the lack of user control over the output style of visualizations. The software does not allow users to customize the look and feel of their visualizations, making it difficult to present the data in a visually appealing and engaging way. 
Review 3: At my team we use Tableau as a data exploration tool for data practitioners, and not for dashboards and data reporting to the company. This setup takes advantage of what Tableau does best and don't use it for what it doesn't.  Tableau as a data reporting and dashboard tool may be suited for more traditional organizations, with a lot of experienced BI personnel who can take the time to build good dashboards with it. 
PROS:Tableau allows for very good exploratory data analysis, and one can build amazing visualizations with it. Once you learn how to work with it, it becomes very easy to explore data and get insights from it. 
CONS:Tableau is very expensive compared to its competitors, and if you want to get the whole company using it for dashboards, the pricing is certainly prohibitive for startups.  I also don't Tableau for Dashboards, as the discovery and navigation interface is not very inviting  to business users and dashboards are hard do discover. At the same time, building dashboards on Tableau is a lot less straightforward than in other tools. One can build very advanced dashboards, but that requires a lot of time and expertise, which not all Data users in a company will be. 
Review 4: Overall, Tableau is an excellent data visualization tool that is incredibly powerful and easy to use. It is a great tool for both novice and experienced users due to its drag-and-drop interface and wide range of features. However, it can be quite pricey and lack some of the advanced analytics capabilities of its competitors.
PROS:Tableau is an incredibly powerful and user-friendly data visualization tool. It is incredibly easy to use and provides incredibly powerful visualizations that can help users better understand their data. The drag-and-drop interface makes it incredibly easy to build complex visualizations with minimal effort. Additionally, Tableau has a wide variety of features that give users the ability to customize their visualizations to their needs.
CONS:One of the drawbacks of Tableau is that it can be quite expensive for larger organizations. Additionally, the user interface is not as intuitive as some of its competitors, which can make it difficult for new users to learn. Finally, Tableau does not have the ability to perform advanced analytics, which can be a limitation for some users.Reasons for switching to TableauTableau stands out from its competitors due to its incredibly user-friendly drag-and-drop interface and wide range of features. Additionally, Tableau's visualizations are incredibly powerful and can be easily customized to fit the user's needs. Finally, Tableau has a large community of users and developers who are constantly creating and sharing new visualizations and resources.tableau Alternatives: Bitrix24, Xtraction, BOARD, InsightSquared, TARGIT.
tableau MAJOR PROS: Fantastic user interface. The data visualization functionality is very powerful and really helps tell the stories behind the data., Its greatest strength is its flexibility to accept almost any kind of data and form them into displays with each other., I use this on a daily basis to check the stats of my surveys for my job. I love how convenient it is to see how great of a job I am doing for customers.
tableau MAJOR CONS: There are some limitations (eg. email alert layout is extremely limited, you can only drag the date range filter as opposed to selecting dates). Some of our dashboards load very slowly., Tableau does lack functionality and is hard to use. There are scaling issues., The support we pay for is almost worthless. We frequently solve the problem ourselves and it is time consuming to submit everything via their helpdesk with nothing to show for it.


alteryx Description: Alteryx is the launchpad for automation breakthroughs. Be it your personal growth, achieving transformative digital outcomes, or rapid innovation, the results are unparalleled. The unique innovation that converges analytics, data science and process automation into one easy-to-use platform, empowers everyone and every organization ?to make business-altering breakthroughs the new status quo.?.
alteryx Overall Rating: 4.78.
alteryx Ease-of-use Rating: 4.5
alteryx Customer Support Rating: 4.5
alteryx Value for money Rating: 4
alteryx Functionality Rating: 4.5
alteryx
Review 1: I can't really discuss use cases due to proprietary information, but let me tell you- Alteryx elevates your entire organization and is a extremely versatile product to add to an analysts toolkit.
PROS:Alteryx is an incredibly easy to use tool that creates and automates data workflows with a click and drag toolbox. Passed R, Python and JSON code through the workflow so you can get technical tasks done without a developer.
CONS:Alteryx support could be a bit better in terms of follow up with very specific and technical questions. However, the community is a great resource to get answers to high level questions!
Review 2: Overall, Alteryx Designer is an excellent piece of software.  Very well-designed!  Previously, I had used SAS Enterprise Guide for similar data cleansing/ETL processing and that software was a complete trainwreck.  It was very clunky in every way and ran very unreliably.  But Alteryx has taken the same concept of building out data processes, step by step, visually (which makes it very easy to tweak later on), only they have all of the best infrastructure to support reliable runtimes, etc.  Using Alteryx Designer is a breeze and I could not be more satisfied with this software solution!  
PROS:I love how simple it is to cleanse data in Alteryx.  You can build linear workflows using their myriad of s.  I love how each  has a built-in tutorial example that will show you how to use the function.  I also really love how you someone with limited SQL knowledge (my colleagues) could use Alteryx to import and refine data to get the exact results they desire.  It really is a very intuitive software--well-documented.  Oh, and their support is FANTASTIC!  I received an email offer to work with an expert from the company and was immediately paired up with a few company trainers who offered their own customized expertise for the specific problems I was working to try to solve.  The online portal is excellent as well; there's a whole community ready and willing to assist and help you get better at using Alteryx.  Very impressive software.  
CONS:The parts I found most difficult to use were the neural networking/forecasting & macro/automation widgets & tools.  The forecasting/neural networking wasn't very straightforward and seemed to really be more of an art than a science.  I would have expected there to be more tutorials on how to effectively use these tools, but I think you would need to have a degree in these arts to really master them.  We are probably still too early into this new science.  The macro & automation functions really blew my mind and I never learned how to use them--for example, to loop through processes.  I would have liked to, but I just didn't have the patience to figure it out on my own and it wasn't high enough of a priority.  
Review 3: 
PROS:It does not require coding skills like python i.e, easier to get started with and do basic data analysis in matter of minutes
CONS:Once a workflow is created, it cannot be viewed or executed by anyone who does not posses a copy of alteryx software which is expensive to buy just for execution of workflows. However, there is a separate product which is called alteryx server where workflows can be executed free of cost. The biggest problem is that behaviour of workflows on designer and server is different. Many tools do not work on server and there is no way of showing intermediate results on server. As these are two different products, one can't even file a ticket of improvement on designer's customer support. In complex workflows things get even more frustrating when server is unable to do what alteryx workflow suppose to do.
Review 4: This software, especially the predictive analytics piece, gave me new insight into my data.  I saved time during the process of merging and cleansing data that could then be put to better use exploring, analyzing, and making decisions with the data.  The work that previously required collaboration between several people to complete could be done by one person in a fraction of the time, which benefited everyone.
PROS:Alteryx has the full package starting with excellent customer service and continuing with an straightforward drag-and-drop interface.  An abundance of available nodes make it simple to pull in data from many different sources, merge, union, filter, sort, and even perform statistical or predictive analysis on your data, and then disseminate it to the proper stakeholders.  The visual nature of the product makes it accessible, even for beginners with no programming background, so they can start building queries right away.  What I love most is how it makes the part of my job that used to be the most tedious--data prep and cleansing--fast and effortless and even fun.  There are also many online training videos and community group to help build your knowledge of the product and its full range of capabilities.
CONS:Alteryx may be better suited to fields other than education; however, I can still find good use cases for it in my job.  
Review 5: 
PROS:1. Drag and drop based UI interface  2. Ability to import data from and export data into variety of data sources  3. Re-usuability of workflows  4. Full feature trial license
CONS:1. Lack of (or limited) visualisation capability.  2.machine dependent licensed(it should had license key Linked to email ID, making it easier to port licensed from one system to another)  3. Expensive licensealteryx Alternatives: Bitrix24, Xtraction, BOARD, Tableau, InsightSquared.
alteryx MAJOR PROS: Great assistance from Altyrex - committed resources at no cost to help implement, train etc. A lot of willingness to help us 'prove' the business case - i.e. make a success of the pilot project., The ease of use to prep, blend, transform, wrangle, etc is awesome. The learning curve is not that challenging., This software is great for visual learners/analysts. It breaks down the ETL process visually, allowing users to understand what processes are at play quickly and easily.
alteryx MAJOR CONS: Perhaps having a pop up to show where the error lies in your work., Costly license limits the use. Requires training for getting used to it., Nothing much I dislike, the only thing I disliked is that we have to use visual tool to view data every time, that slows the running time of the workflow.

microsoft-power-bi Description: Microsoft Power BI is a web-based business analytics and data visualization platform that is suitable for businesses of all sizes. It monitors important organizational data and also from all apps used by organizations. Microsoft Power BI provides tools to quickly analyze, transform and visualize data, and also share reports. Microsoft Power BI offers SQL Server Analysis Services through which users can quickly build reusable models using the overall data. The software enables users to integrate their apps, so as to deliver reports along with real-time dashboards. Microsoft Power BI also provides self-service access to major third-party cloud sources such as GitHub, Zendesk, Marketo and Salesforce. Microsoft Power BI is available both in free and paid (Pro) versions. 
microsoft-power-bi Pricing Details: Starting price: $9.99 per month, Free trial: Available, Free version: Not Available.
microsoft-power-bi Overall Rating: 4.55.
microsoft-power-bi Ease-of-use Rating: 4
microsoft-power-bi Customer Support Rating: 4
microsoft-power-bi Value for money Rating: 4.5
microsoft-power-bi Functionality Rating: 4.5
microsoft-power-bi
Review 1: Power BI allows me to manipulate data and present it in visually appealing and flexible formats.
PROS:I have used Tableau and Google Data Studio - among other alternatives. Nothing has seemed to compare to the functionality that Power BI offers. Everything about the program is made with the full spectrum of analysis in mind - from the designer to the data 'wrangler' to the end consumer (ie. others within the enterprise). In this way, Power BI is extremely powerful because it can be used for so many different types of projects; the data-analysis intensive to the more visual 'data-presentation' dashboards.
CONS:Sharing dashboards and reports in Power BI is easy and it has a lot of options for doing so. However, the requirement that all users must have a Pro Account to view dashboards created in Pro is a bit of a hindrance in a global organization where it does not make economical sense for every employee to have a Pro account.  It seems like it would be a great addition to allow limited sharing to everyone - even those without a Pro account, as most of my internal team does not have an account.Reasons for switching to Microsoft Power BIData Studio has limited functionality, which is understandable since it is free. Power BI has more built-in data analysis and data processing abilities.
Review 2: Having used Power BI for the past two years, I have developed an in-depth understanding of its capabilities and features, which include various data connectors, visualization options, and modeling tools. I have also gained proficiency in extracting meaningful insights from data by leveraging the tool's powerful data analysis functionalities.As a Power BI user with two years of experience, I value the tool's ability to effortlessly connect and integrate data from diverse sources, enabling the creation of comprehensive dashboards that provide insights into various aspects of our business. Additionally, the ability to customize and share these dashboards with others in our organization has proven invaluable in facilitating better decision-making processes.Overall, my experience with Power BI has provided me with a significant advantage in analyzing data and gaining insights into various aspects of our business, which has ultimately contributed to improved performance and outcomes.
PROS:Microsoft Power BI is a powerful tool for business analytics, offering a diverse range of features such as data visualization, modeling, and analysis. Its user-friendly interface, wide selection of data connectors, and ability to handle large volumes of data are some of the reasons why I appreciate it. Additionally, value its flexibility in terms of customization and integration with other Microsoft products
CONS:Sometimes this power bi desktop is not responding when the data is loading.
Review 3: I am generally satisficed  with the product, price and ease of learning with lots of YouTube video  and the Power BI community, there is always something new to learn and implemented to improve or help meet your reporting requirements.
PROS:Easy to get started with Power BI if you are on the Microsoft 365 platform. Power Bi provides an easy to connect and interactive method for pulling data in real time from data collection system such as the SharePoint online, MS Excel and others. The ease of connecting with SharePoint is the strong point for my review. Having the Power Bi connected to the SharePoint system provides the opportunity to have real time dashboards and dynamic reports for any small organization that is wishing to provide real or near to real-time reporting without the heavy upfront implementation cost for software, training and human resource.  Microsoft has indeed liberated the Dynamic Reporting with Power BI, and has also allowed for some very elegant web reporting to be created by non technical users. My simple work flow , SharePoint for data collection, PowerBi for reporting.  
CONS:The main issues are around the changes that are made by Microsoft, this will always be a learning curve for the users. The lack of consistency  on certain actions will be the biggest issue. Reason for choosing Microsoft Power BICost, Familiarity and given that is is a free offer within the MS Offfice 365 or M365 platform, it leads to an easy choice. 
Review 4: Overall, Microsoft PowerBI is a well-known business intelligence application that is extensively utilised by businesses of all kinds. Its robust data visualisation features, data connection choices, and capacity to find insights using AI and machine learning make it a useful tool for organisations looking to extract more value from their data. However, for certain users, the steep learning curve, restricted data preparation capabilities, and convoluted licencing approach may be disadvantages. Furthermore, the restricted third-party connection may be inconvenient for enterprises that employ a range of tools and systems. Regardless of these restrictions, PowerBI is a highly rated programme that is extensively used and trusted by enterprises seeking a comprehensive business intelligence solution.
PROS:PowerBI enables users to connect to a wide range of data sources and generate visually beautiful visuals that bring data to life and make it easier to comprehend. The programme is also extremely customisable, allowing customers to design dashboards and reports that are tailored to their individual requirements. PowerBI's mobile capability and connection with Microsoft products like as Excel make it an excellent alternative for enterprises that require on-the-go data access and interaction. Furthermore, PowerBI's AI and machine learning features enable users to quickly uncover insights and trends in their data, making it a useful tool for businesses of all sizes.
CONS:PowerBI has a steep learning curve, making it difficult for new users to rapidly become acquainted with the product. For some users, particularly those new to business intelligence and data analysis, this might be a barrier to entrance. Because PowerBI lacks a powerful data preparation tool, users may find it difficult to clean, process, and prepare their data for analysis. This might be a disadvantage for those that require more complex data preparation operations.microsoft-power-bi Alternatives: Xtraction, TARGIT, Style Intelligence, ClicData, Dundas BI.
microsoft-power-bi MAJOR PROS: Perfect integration with Microsoft solutions. Ease to share with other partners., In my opinion, it helps its users create very professional-looking and most importantly dynamic reports with stunning visuals., The brightest part of Power BI for us has been the ability to create rich data driven dashboards without having to invest in an entire new platform..
microsoft-power-bi MAJOR CONS: I have trouble connecting it to an analysis cube and making a website version., Some find it too complicated to use and are intimidated by the format. I have not been able to find training for management on how to use a report that someone else with PowerBI skills created., It has lot of learning curve and has very limited support available on internet. Configuration is a hell." }); Set(outvar1, ""); Collect(wordtemplatelitm, [{ templatename : "Task" },{ templatename : "Context" },{ templatename : "Summary" }]);Collect(style1, [{ stylename : "Precise" },{ stylename : "Balance" },{ stylename : "Creative" }]);
```

- add a button to send the chat to chatgpt
- Allow the users to ask any question on data

- Add another dropdown for style
- Assign style1 collection to dropdown

```
["Balance","Precise", "Creative"]
```

- Here is the code for button to send to chatgpt to get information

```
Collect(convlist1, { role : "user", content: Concatenate(TextInput6.Text, " your response should be ", Dropdown2.SelectedText.Value ," add followup suggestions") } );Set(outputvar1, rfpchatgptprocessing.Run(JSON(convlist1)));Set(outvar1, Text(ParseJSON(outputvar1.output).choices.'0'.message.content));Collect(convlist1, { role : "assistant", content: outvar1});
```

- Add check box and drop down for gallery
- For drop down and for items please use this

```
["NA","Task", "Context", "Summary"]
```

- Add another button
- code for button

```
Set(summarytext6, PopulateWord1.Run(JSON(ForAll(Filter(Gallery2.AllItems, (Checkbox2.Value)), Concatenate(role, "-", content, "-", Checkbox2.Value, "-", Dropdown1.Selected.Value)), JSONFormat.IgnoreUnsupportedTypes & JSONFormat.IgnoreBinaryData)))
```

- add a delete X button to delete the chat

```
Remove(convlist1,ThisItem);
```

- on the arrow button add this code

```
Navigate( EditForm )
```

- Create a new Edit form
- Create a Edit form
- Assign the data source to the convlist1
- Add a data card
- For Item add

```
ThisItem.content
```

- Add a text input
- assign the text input to the data card
- Make the text box multiline and bigger

```
ThisItem.content
```

- Bring the save button

```
Patch(convlist1,{content:ThisItem.content},{content:TextInput10.Text})
```

- Idea here will be ability to pick the response and edit and then select them to process into Word Template
- When you select the response it will be added to the gallery and then select the item and create RFP word document
- Now add a clear button to revise the chat
- copy the same from onvisible property

```
Clear(convlist1);Collect(convlist1, { role : "system", content: "Answer the question based on the context below. Keep the answer short and concise. Provide followup questions: Tableau is an integrated business intelligence (BI) and analytics solution that helps to analyze key business data and generate meaningful insights. The solution helps businesses to collect data from multiple source points such as SQL databases, spreadsheets, cloud apps like Google Analytics and Salesforce to create a collective dataset. Tableau's live visual analytics and interactive dashboard allow slicing & dicing datasets for generating relevant insights and exploring new opportunities. Users can create interactive maps and analyze data across regions, territories, demographics and more. Tableau helps to create a narrative story of the data analysis with interactive visualizations that can be shared with their audience. Tableau can be customized to serve a
tableau Overall Rating: 4.55.
tableau Ease-of-use Rating: 4
tableau Customer Support Rating: 4
tableau Value for money Rating: 4
tableau Functionality Rating: 4.5
tableau
Review 1: I have worked on domains like Telecom and Network where in I have been using Tableau. Its really liked by senior management as it shows each and every bit very clearly to relevant stake holders. For example, Marketing head likes region wise revenue sale dashboard which gives region/sales manager/product wise sale.
PROS:There are many features which lot of users would have liked:- 1. Connection to almost all database 2. Many types of Charts can be created including Sankey, Doughnut, Maps etc. 3. Its has the feature of data blending  and data extraction 4. Dashboards can be viewed on multiple devices like tablet, mobile and laptops. 5. Its really fast while drilling down or filtering out from any dashboard.
CONS:Its a bit expensive tool as compared to other data visualization tools however its a value for money.Reason for choosing TableauThere were multiple check points before choosing Tableau:- 1. We refereed Gartner quadrant which showcased good image of tableau. 2. Reviews taken from friends who are working in other organisation. 3. Some POCs developed on Trail version of TableauReasons for switching to TableauIt was a organisation demand however but as and when I started working on Tableau, I really liked working on it.
Review 2: I believe Tableau is an excellent data visualization tool that is easy to use and provides powerful features for creating interactive visualizations and dashboards quickly.
PROS:Tableau is an outstanding data visualization tool that makes data analysis simple and straightforward. It offers an intuitive drag-and-drop user interface and powerful workbook features to let users develop interactive visualizations and dashboards quickly. I particularly admire Tableau's ability to create appealing visualizations effortlessly. It lets users generate an array of charts, graphs, and maps that allow them to deliver the data's story in no time. Tableau also gives users access to a wide range of analytical abilities like forecasting, trends, and correlations. It is also convenient to share visualizations with the public, enabling users to build interactive web-based visualizations with ease. To top it off, Tableau has a formidable suite of tools for data preparation which allows users to promptly clean, structure, and format data for analysis.
CONS:One of the least liked features of Tableau is the lack of user control over the output style of visualizations. The software does not allow users to customize the look and feel of their visualizations, making it difficult to present the data in a visually appealing and engaging way. 
Review 3: At my team we use Tableau as a data exploration tool for data practitioners, and not for dashboards and data reporting to the company. This setup takes advantage of what Tableau does best and don't use it for what it doesn't.  Tableau as a data reporting and dashboard tool may be suited for more traditional organizations, with a lot of experienced BI personnel who can take the time to build good dashboards with it. 
PROS:Tableau allows for very good exploratory data analysis, and one can build amazing visualizations with it. Once you learn how to work with it, it becomes very easy to explore data and get insights from it. 
CONS:Tableau is very expensive compared to its competitors, and if you want to get the whole company using it for dashboards, the pricing is certainly prohibitive for startups.  I also don't Tableau for Dashboards, as the discovery and navigation interface is not very inviting  to business users and dashboards are hard do discover. At the same time, building dashboards on Tableau is a lot less straightforward than in other tools. One can build very advanced dashboards, but that requires a lot of time and expertise, which not all Data users in a company will be. 
Review 4: Overall, Tableau is an excellent data visualization tool that is incredibly powerful and easy to use. It is a great tool for both novice and experienced users due to its drag-and-drop interface and wide range of features. However, it can be quite pricey and lack some of the advanced analytics capabilities of its competitors.
PROS:Tableau is an incredibly powerful and user-friendly data visualization tool. It is incredibly easy to use and provides incredibly powerful visualizations that can help users better understand their data. The drag-and-drop interface makes it incredibly easy to build complex visualizations with minimal effort. Additionally, Tableau has a wide variety of features that give users the ability to customize their visualizations to their needs.
CONS:One of the drawbacks of Tableau is that it can be quite expensive for larger organizations. Additionally, the user interface is not as intuitive as some of its competitors, which can make it difficult for new users to learn. Finally, Tableau does not have the ability to perform advanced analytics, which can be a limitation for some users.Reasons for switching to TableauTableau stands out from its competitors due to its incredibly user-friendly drag-and-drop interface and wide range of features. Additionally, Tableau's visualizations are incredibly powerful and can be easily customized to fit the user's needs. Finally, Tableau has a large community of users and developers who are constantly creating and sharing new visualizations and resources.tableau Alternatives: Bitrix24, Xtraction, BOARD, InsightSquared, TARGIT.
tableau MAJOR PROS: Fantastic user interface. The data visualization functionality is very powerful and really helps tell the stories behind the data., Its greatest strength is its flexibility to accept almost any kind of data and form them into displays with each other., I use this on a daily basis to check the stats of my surveys for my job. I love how convenient it is to see how great of a job I am doing for customers.
tableau MAJOR CONS: There are some limitations (eg. email alert layout is extremely limited, you can only drag the date range filter as opposed to selecting dates). Some of our dashboards load very slowly., Tableau does lack functionality and is hard to use. There are scaling issues., The support we pay for is almost worthless. We frequently solve the problem ourselves and it is time consuming to submit everything via their helpdesk with nothing to show for it.


alteryx Description: Alteryx is the launchpad for automation breakthroughs. Be it your personal growth, achieving transformative digital outcomes, or rapid innovation, the results are unparalleled. The unique innovation that converges analytics, data science and process automation into one easy-to-use platform, empowers everyone and every organization ?to make business-altering breakthroughs the new status quo.?.
alteryx Overall Rating: 4.78.
alteryx Ease-of-use Rating: 4.5
alteryx Customer Support Rating: 4.5
alteryx Value for money Rating: 4
alteryx Functionality Rating: 4.5
alteryx
Review 1: I can't really discuss use cases due to proprietary information, but let me tell you- Alteryx elevates your entire organization and is a extremely versatile product to add to an analysts toolkit.
PROS:Alteryx is an incredibly easy to use tool that creates and automates data workflows with a click and drag toolbox. Passed R, Python and JSON code through the workflow so you can get technical tasks done without a developer.
CONS:Alteryx support could be a bit better in terms of follow up with very specific and technical questions. However, the community is a great resource to get answers to high level questions!
Review 2: Overall, Alteryx Designer is an excellent piece of software.  Very well-designed!  Previously, I had used SAS Enterprise Guide for similar data cleansing/ETL processing and that software was a complete trainwreck.  It was very clunky in every way and ran very unreliably.  But Alteryx has taken the same concept of building out data processes, step by step, visually (which makes it very easy to tweak later on), only they have all of the best infrastructure to support reliable runtimes, etc.  Using Alteryx Designer is a breeze and I could not be more satisfied with this software solution!  
PROS:I love how simple it is to cleanse data in Alteryx.  You can build linear workflows using their myriad of s.  I love how each  has a built-in tutorial example that will show you how to use the function.  I also really love how you someone with limited SQL knowledge (my colleagues) could use Alteryx to import and refine data to get the exact results they desire.  It really is a very intuitive software--well-documented.  Oh, and their support is FANTASTIC!  I received an email offer to work with an expert from the company and was immediately paired up with a few company trainers who offered their own customized expertise for the specific problems I was working to try to solve.  The online portal is excellent as well; there's a whole community ready and willing to assist and help you get better at using Alteryx.  Very impressive software.  
CONS:The parts I found most difficult to use were the neural networking/forecasting & macro/automation widgets & tools.  The forecasting/neural networking wasn't very straightforward and seemed to really be more of an art than a science.  I would have expected there to be more tutorials on how to effectively use these tools, but I think you would need to have a degree in these arts to really master them.  We are probably still too early into this new science.  The macro & automation functions really blew my mind and I never learned how to use them--for example, to loop through processes.  I would have liked to, but I just didn't have the patience to figure it out on my own and it wasn't high enough of a priority.  
Review 3: 
PROS:It does not require coding skills like python i.e, easier to get started with and do basic data analysis in matter of minutes
CONS:Once a workflow is created, it cannot be viewed or executed by anyone who does not posses a copy of alteryx software which is expensive to buy just for execution of workflows. However, there is a separate product which is called alteryx server where workflows can be executed free of cost. The biggest problem is that behaviour of workflows on designer and server is different. Many tools do not work on server and there is no way of showing intermediate results on server. As these are two different products, one can't even file a ticket of improvement on designer's customer support. In complex workflows things get even more frustrating when server is unable to do what alteryx workflow suppose to do.
Review 4: This software, especially the predictive analytics piece, gave me new insight into my data.  I saved time during the process of merging and cleansing data that could then be put to better use exploring, analyzing, and making decisions with the data.  The work that previously required collaboration between several people to complete could be done by one person in a fraction of the time, which benefited everyone.
PROS:Alteryx has the full package starting with excellent customer service and continuing with an straightforward drag-and-drop interface.  An abundance of available nodes make it simple to pull in data from many different sources, merge, union, filter, sort, and even perform statistical or predictive analysis on your data, and then disseminate it to the proper stakeholders.  The visual nature of the product makes it accessible, even for beginners with no programming background, so they can start building queries right away.  What I love most is how it makes the part of my job that used to be the most tedious--data prep and cleansing--fast and effortless and even fun.  There are also many online training videos and community group to help build your knowledge of the product and its full range of capabilities.
CONS:Alteryx may be better suited to fields other than education; however, I can still find good use cases for it in my job.  
Review 5: 
PROS:1. Drag and drop based UI interface  2. Ability to import data from and export data into variety of data sources  3. Re-usuability of workflows  4. Full feature trial license
CONS:1. Lack of (or limited) visualisation capability.  2.machine dependent licensed(it should had license key Linked to email ID, making it easier to port licensed from one system to another)  3. Expensive licensealteryx Alternatives: Bitrix24, Xtraction, BOARD, Tableau, InsightSquared.
alteryx MAJOR PROS: Great assistance from Altyrex - committed resources at no cost to help implement, train etc. A lot of willingness to help us 'prove' the business case - i.e. make a success of the pilot project., The ease of use to prep, blend, transform, wrangle, etc is awesome. The learning curve is not that challenging., This software is great for visual learners/analysts. It breaks down the ETL process visually, allowing users to understand what processes are at play quickly and easily.
alteryx MAJOR CONS: Perhaps having a pop up to show where the error lies in your work., Costly license limits the use. Requires training for getting used to it., Nothing much I dislike, the only thing I disliked is that we have to use visual tool to view data every time, that slows the running time of the workflow.

microsoft-power-bi Description: Microsoft Power BI is a web-based business analytics and data visualization platform that is suitable for businesses of all sizes. It monitors important organizational data and also from all apps used by organizations. Microsoft Power BI provides tools to quickly analyze, transform and visualize data, and also share reports. Microsoft Power BI offers SQL Server Analysis Services through which users can quickly build reusable models using the overall data. The software enables users to integrate their apps, so as to deliver reports along with real-time dashboards. Microsoft Power BI also provides self-service access to major third-party cloud sources such as GitHub, Zendesk, Marketo and Salesforce. Microsoft Power BI is available both in free and paid (Pro) versions. 
microsoft-power-bi Pricing Details: Starting price: $9.99 per month, Free trial: Available, Free version: Not Available.
microsoft-power-bi Overall Rating: 4.55.
microsoft-power-bi Ease-of-use Rating: 4
microsoft-power-bi Customer Support Rating: 4
microsoft-power-bi Value for money Rating: 4.5
microsoft-power-bi Functionality Rating: 4.5
microsoft-power-bi
Review 1: Power BI allows me to manipulate data and present it in visually appealing and flexible formats.
PROS:I have used Tableau and Google Data Studio - among other alternatives. Nothing has seemed to compare to the functionality that Power BI offers. Everything about the program is made with the full spectrum of analysis in mind - from the designer to the data 'wrangler' to the end consumer (ie. others within the enterprise). In this way, Power BI is extremely powerful because it can be used for so many different types of projects; the data-analysis intensive to the more visual 'data-presentation' dashboards.
CONS:Sharing dashboards and reports in Power BI is easy and it has a lot of options for doing so. However, the requirement that all users must have a Pro Account to view dashboards created in Pro is a bit of a hindrance in a global organization where it does not make economical sense for every employee to have a Pro account.  It seems like it would be a great addition to allow limited sharing to everyone - even those without a Pro account, as most of my internal team does not have an account.Reasons for switching to Microsoft Power BIData Studio has limited functionality, which is understandable since it is free. Power BI has more built-in data analysis and data processing abilities.
Review 2: Having used Power BI for the past two years, I have developed an in-depth understanding of its capabilities and features, which include various data connectors, visualization options, and modeling tools. I have also gained proficiency in extracting meaningful insights from data by leveraging the tool's powerful data analysis functionalities.As a Power BI user with two years of experience, I value the tool's ability to effortlessly connect and integrate data from diverse sources, enabling the creation of comprehensive dashboards that provide insights into various aspects of our business. Additionally, the ability to customize and share these dashboards with others in our organization has proven invaluable in facilitating better decision-making processes.Overall, my experience with Power BI has provided me with a significant advantage in analyzing data and gaining insights into various aspects of our business, which has ultimately contributed to improved performance and outcomes.
PROS:Microsoft Power BI is a powerful tool for business analytics, offering a diverse range of features such as data visualization, modeling, and analysis. Its user-friendly interface, wide selection of data connectors, and ability to handle large volumes of data are some of the reasons why I appreciate it. Additionally, value its flexibility in terms of customization and integration with other Microsoft products
CONS:Sometimes this power bi desktop is not responding when the data is loading.
Review 3: I am generally satisficed  with the product, price and ease of learning with lots of YouTube video  and the Power BI community, there is always something new to learn and implemented to improve or help meet your reporting requirements.
PROS:Easy to get started with Power BI if you are on the Microsoft 365 platform. Power Bi provides an easy to connect and interactive method for pulling data in real time from data collection system such as the SharePoint online, MS Excel and others. The ease of connecting with SharePoint is the strong point for my review. Having the Power Bi connected to the SharePoint system provides the opportunity to have real time dashboards and dynamic reports for any small organization that is wishing to provide real or near to real-time reporting without the heavy upfront implementation cost for software, training and human resource.  Microsoft has indeed liberated the Dynamic Reporting with Power BI, and has also allowed for some very elegant web reporting to be created by non technical users. My simple work flow , SharePoint for data collection, PowerBi for reporting.  
CONS:The main issues are around the changes that are made by Microsoft, this will always be a learning curve for the users. The lack of consistency  on certain actions will be the biggest issue. Reason for choosing Microsoft Power BICost, Familiarity and given that is is a free offer within the MS Offfice 365 or M365 platform, it leads to an easy choice. 
Review 4: Overall, Microsoft PowerBI is a well-known business intelligence application that is extensively utilised by businesses of all kinds. Its robust data visualisation features, data connection choices, and capacity to find insights using AI and machine learning make it a useful tool for organisations looking to extract more value from their data. However, for certain users, the steep learning curve, restricted data preparation capabilities, and convoluted licencing approach may be disadvantages. Furthermore, the restricted third-party connection may be inconvenient for enterprises that employ a range of tools and systems. Regardless of these restrictions, PowerBI is a highly rated programme that is extensively used and trusted by enterprises seeking a comprehensive business intelligence solution.
PROS:PowerBI enables users to connect to a wide range of data sources and generate visually beautiful visuals that bring data to life and make it easier to comprehend. The programme is also extremely customisable, allowing customers to design dashboards and reports that are tailored to their individual requirements. PowerBI's mobile capability and connection with Microsoft products like as Excel make it an excellent alternative for enterprises that require on-the-go data access and interaction. Furthermore, PowerBI's AI and machine learning features enable users to quickly uncover insights and trends in their data, making it a useful tool for businesses of all sizes.
CONS:PowerBI has a steep learning curve, making it difficult for new users to rapidly become acquainted with the product. For some users, particularly those new to business intelligence and data analysis, this might be a barrier to entrance. Because PowerBI lacks a powerful data preparation tool, users may find it difficult to clean, process, and prepare their data for analysis. This might be a disadvantage for those that require more complex data preparation operations.microsoft-power-bi Alternatives: Xtraction, TARGIT, Style Intelligence, ClicData, Dundas BI.
microsoft-power-bi MAJOR PROS: Perfect integration with Microsoft solutions. Ease to share with other partners., In my opinion, it helps its users create very professional-looking and most importantly dynamic reports with stunning visuals., The brightest part of Power BI for us has been the ability to create rich data driven dashboards without having to invest in an entire new platform..
microsoft-power-bi MAJOR CONS: I have trouble connecting it to an analysis cube and making a website version., Some find it too complicated to use and are intimidated by the format. I have not been able to find training for management on how to use a report that someone else with PowerBI skills created., It has lot of learning curve and has very limited support available on internet. Configuration is a hell." });Set(outvar1, "");
Reset(TextInput5);
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AzureAI/images/rfpdoc16.jpg "Architecture")