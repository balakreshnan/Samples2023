# Building LLM Application using Azure Open AI and HA, DR

## Introduction

- Design HA and DR architecture for LLM application
- Inside same region
- Across regions
- Across globally
- Using Azure Open AI
- Using App service for semantic kernel or web application
- Using Azure Cognitive Search for Vector Index.
- Implementing RAG - Retrieval Augumented Generation

## Architecture

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/AOAI/images/ragarchsimpleHADR.jpg "Architecture")

## Architecture - Explained

- Azure Open AI in mutiple regions
- Put a Application Gateway in front of it, for load balancing and HA
- You can also keep the application local to the open AI region
- Search and web application locally in the region for the application
- All the web application and search, use APIM for HA and DR
- APIM can be deployed in multiple regions
- Cosmos DB can be replicated across regions
- To have best latency between the application keep all components in same region
- Networking is up to you, you can use VNET peering or VPN or Express Route
- Other security controls is up to enterprise and their standards and process
- Also use Azure Monitor to monitor the application and the infrastructure
- Most of the inter application communication can be managed identity or other authentication mechanism
- Most of them use HTTPS as their protocol for communication