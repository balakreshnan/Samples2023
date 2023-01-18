# Multivariant time series anomaly detection using Spark with SynapseML

## Introduction

- Using synanpse ML to detect anomalies in multivariant time series data
- We are using Azure syanpse analytics spark enginer 3.2
- install synapseML package from - https://microsoft.github.io/SynapseML/docs/features/cognitive_services/CognitiveServices%20-%20Multivariate%20Anomaly%20Detection/
- Please follow the above link for current implementation details
- Variations in package versions may cause issues

## Requirements

- Create a Anomaly detection cognitive service in Azure portal
- Note down the key and endpoint
- Save the key in Azure KeyVault
- Location is needed
- Create a storage account to store temporary files
- Also store input data in the storage account
- Store output in storage account
- Create Spark pool with Spark version 3.2
- Data is available in the storage account - wasbs://publicwasb@mmlspark.blob.core.windows.net/MVAD/sample.csv

## Code

- Create a notebook in Azure Synapse Analytics
- install synapseML package

```
%%configure -f
{
  "name": "synapseml",
  "conf": {
      "spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.10.2",
      "spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
      "spark.jars.excludes": "org.scala-lang:scala-reflect,org.apache.spark:spark-tags_2.12,org.scalactic:scalactic_2.12,org.scalatest:scalatest_2.12,com.fasterxml.jackson.core:jackson-databind",
      "spark.yarn.user.classpath.first": "true"
  }
}
```

- Print synapseml version

```
import synapse.ml.core

synapse.ml.core.__spark_package_version__
```

- import required packages

```
from synapse.ml.cognitive import *
from notebookutils import mssparkutils
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType
import synapse.ml
import matplotlib.pyplot as plt

from synapse.ml.cognitive import *
```

```
import os
from pyspark.sql import SparkSession
from synapse.ml.core.platform import find_secret

# Bootstrap Spark Session
spark = SparkSession.builder.getOrCreate()
```

- get azure keyvault secrets

```
anomalyKey = mssparkutils.credentials.getSecret("mlopskeyv1","anomalykey")
```

- setup parameters

```
anomalyKey = mssparkutils.credentials.getSecret("mlopskeyv1","anomalykey")
# Your storage account name
storageName = "storagename"
# A connection string to your blob storage account
#storageKey = find_secret("madtest-storage-key")
storageKey = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# A place to save intermediate MVAD results
intermediateSaveDir = (
    "wasbs://metricfolder@synpasedlstore.blob.core.windows.net/intermediateData"
)
# The location of the anomaly detector resource that you created
location = "eastus"
```

- set the storage configuration

```
spark.sparkContext._jsc.hadoopConfiguration().set(
    f"fs.azure.account.key.{storageName}.blob.core.windows.net", storageKey
)
```

- load the data into dataframe

```
df = (
    spark.read.format("csv")
    .option("header", "true")
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/MVAD/sample.csv")
)
```

- make column data type change to double

```
df = (
    df.withColumn("sensor_1", col("sensor_1").cast(DoubleType()))
    .withColumn("sensor_2", col("sensor_2").cast(DoubleType()))
    .withColumn("sensor_3", col("sensor_3").cast(DoubleType()))
)

# Let's inspect the dataframe:
df.show(5)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/multivarts1.jpg "Output Episodes")

- Set the MultivariateAnomalyDetector parameters and Estimators

```
trainingStartTime = "2020-06-01T12:00:00Z"
trainingEndTime = "2020-07-02T17:55:00Z"
timestampColumn = "timestamp"
inputColumns = ["sensor_1", "sensor_2", "sensor_3"]

estimator = (
    FitMultivariateAnomaly()
    .setSubscriptionKey(anomalyKey)
    .setLocation(location)
    .setStartTime(trainingStartTime)
    .setEndTime(trainingEndTime)
    .setIntermediateSaveDir(intermediateSaveDir)
    .setTimestampCol(timestampColumn)
    .setInputCols(inputColumns)
    .setSlidingWindow(200)
)
```

- Fit the model

```
model = estimator.fit(df)
```

- now inference

```
inferenceStartTime = "2020-07-02T18:00:00Z"
inferenceEndTime = "2020-07-06T05:15:00Z"

result = (
    model.setStartTime(inferenceStartTime)
    .setEndTime(inferenceEndTime)
    .setOutputCol("results")
    .setErrorCol("errors")
    .setInputCols(inputColumns)
    .setTimestampCol(timestampColumn)
    .transform(df)
)

result.show(5)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/multivarts2.jpg "Output Episodes")

```
rdf = (
    result.select(
        "timestamp",
        *inputColumns,
        "results.contributors",
        "results.isAnomaly",
        "results.severity"
    )
    .orderBy("timestamp", ascending=True)
    .filter(col("timestamp") >= lit(inferenceStartTime))
    .toPandas()
)

rdf
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/multivarts3.jpg "Output Episodes")

- parse the output

```
def parse(x):
    if type(x) is list:
        return dict([item[::-1] for item in x])
    else:
        return {"series_0": 0, "series_1": 0, "series_2": 0}


rdf["contributors"] = rdf["contributors"].apply(parse)
rdf = pd.concat(
    [rdf.drop(["contributors"], axis=1), pd.json_normalize(rdf["contributors"])], axis=1
)
rdf
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/multivarts4.jpg "Output Episodes")

- plot the output

```
minSeverity = 0.1


####### Main Figure #######
plt.figure(figsize=(23, 8))
plt.plot(
    rdf["timestamp"],
    rdf["sensor_1"],
    color="tab:orange",
    line,
    linewidth=2,
    label="sensor_1",
)
plt.plot(
    rdf["timestamp"],
    rdf["sensor_2"],
    color="tab:green",
    line,
    linewidth=2,
    label="sensor_2",
)
plt.plot(
    rdf["timestamp"],
    rdf["sensor_3"],
    color="tab:blue",
    line,
    linewidth=2,
    label="sensor_3",
)
plt.grid(axis="y")
plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
plt.legend()

anoms = list(rdf["severity"] >= minSeverity)
_, _, ymin, ymax = plt.axis()
plt.vlines(np.where(anoms), ymin=ymin, ymax=ymax, color="r", alpha=0.8)

plt.legend()
plt.title(
    "A plot of the values from the three sensors with the detected anomalies highlighted in red."
)
plt.show()

####### Severity Figure #######
plt.figure(figsize=(23, 1))
plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
plt.plot(
    rdf["timestamp"],
    rdf["severity"],
    color="black",
    line,
    linewidth=2,
    label="Severity score",
)
plt.plot(
    rdf["timestamp"],
    [minSeverity] * len(rdf["severity"]),
    color="red",
    line,
    linewidth=1,
    label="minSeverity",
)
plt.grid(axis="y")
plt.legend()
plt.ylim([0, 1])
plt.title("Severity of the detected anomalies")
plt.show()

####### Contributors Figure #######
plt.figure(figsize=(23, 1))
plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
plt.bar(
    rdf["timestamp"], rdf["series_0"], width=2, color="tab:orange", label="sensor_1"
)
plt.bar(
    rdf["timestamp"],
    rdf["series_1"],
    width=2,
    color="tab:green",
    label="sensor_2",
    bottom=rdf["series_0"],
)
plt.bar(
    rdf["timestamp"],
    rdf["series_2"],
    width=2,
    color="tab:blue",
    label="sensor_3",
    bottom=rdf["series_0"] + rdf["series_1"],
)
plt.grid(axis="y")
plt.legend()
plt.ylim([0, 1])
plt.title("The contribution of each sensor to the detected anomaly")
plt.show()
```

- clean up

```
simpleMultiAnomalyEstimator.cleanUpIntermediateData()
model.cleanUpIntermediateData()
```