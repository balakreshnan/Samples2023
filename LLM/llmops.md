# Large Language Model Ops (LLM Ops)

## Introduction

- Create ML Ops for LLM's
- Build end to end development and deployment cycle
- Add Responsible AI to LLM's
- Add Abuse detection to LLM's
- High level process and flow
- LLM Ops is people, process and technology


## LLM Ops flow - Architecture

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/LLM/images/LLMops1.jpg "Architecture")

## Architecture explained

- First it starts with business problem to solve
- Find the data for the problem to solve, this could be a iterative process
- Prompt Engineering - this is where figuring out what is the right prompt to use for the problem
- Develop the LLM application using existing models or train a new model
- Model selection can be based on use case, performance, cost, latency, etc
- Test and validate the prompt engineering and see the output with application is as expected
- This is an iterative pattern
- Add monitoring and auditing code to log prompts and completion
- Also incorporate code for content safety and abuse detection
- Also detect PII and PHI and other sensitive information and log and mask them
- Evaluate and take a decision if the model is ready to move other environments
- Deploy to UAT or staging environment
- Evaluate and refine as needed to make sure application is ready for production
- Deploy to production
- Setup the Monitoring, Auditing and COntent safety system to monitor the application
- If any abuse or content safety issues are detected, then alert the team and take action, mostly human review is needed
- Storage all prompts and completions in a data lake for future use and also metadata about api, configurations etc
- Deploy as Real time or batch endpoint for various applications to consume
- Consumers can be internal or external user or applications