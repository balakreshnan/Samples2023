# Azure Open AI Use your own data - RAG pattern architecture

## Introduction

- Create a gloabally distributed application using Azure services
- Application should be highly available and resilient
- minimal downtime when one of the regions goes down
- Load balanced across regions
- Traffic should be routed to the closest region
- Data should be replicated across regions

## Architecture

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AOAI/images/arch1.jpg "Architecture")

## Architecture Explained

- Azure open ai is created in different subscription in different regions
- Create multiple subscriptions for Azure Open AI
- It's best go across regions locally or globally
- Put a Load balancer in front of the azure open ai within region for multiple subscription
- Put a Traffic manager in front of the azure open ai across regions
- Since there is no data stored, in disaster we can re route the traffic to another subscription or region
- Create Azure cognitive search service in different paired regions
- Create the application in the region where paired regions is available
- Using Azure ML to create LLM Application using prompt flow
- Deploy and manage custom LLM in managed endpoint