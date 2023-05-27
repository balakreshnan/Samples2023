# Microsoft Fabric TPCH copy data to lake house

## Prerequisites

- Microsoft Fabric Account
- Microsoft Fabric Cluster
- Power Admin screen will have options to enable Fabric in a corporate environment

## Steps

- Let's create a pipeline with data flow
- Go to Data Engineering -> Pipelines -> + Create Pipeline
- Give a name to the pipeline
- Drag the copy activity
- Moving the TPCH data from my existing blob storage which are parquet files to the lake house
- Source: Azure Blob Storage
- Sink would be Lake house
- Do binary copy
- Using default configuration for Fabric cluster

## Source

- COnfigure the source

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch3.jpg "Architecture")

- Configuation

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch5.jpg "Architecture")

## Sink

- Configure the sink
- Has to be Lake house inside Fabric

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch4.jpg "Architecture")

- Configuration
- For both source and sink please use no compression
- Save the data in lakehouse called tpchdl

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch6.jpg "Architecture")

- Save and run the pipeline

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch1.jpg "Architecture")

- Details on the copy of data

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch2.jpg "Architecture")

## Spark Processing

- Go to Data science for notebook
- Create a new notebook
- tpchnotebook
- Make sure notebook is connected to lake house called tpchdl
- Read one of the parquet file

```
dfcustomer = spark.read.parquet('Files/CUSTOMER')
display(dfcustomer)
```

- If you an view the data then we have succesfully copied the data from blob to lake house