# Azure Syanpse analytics - Data Flow and Syanpse Spark

## Using Dataflow and Syanpse Spark to analyze data and Spark ML modelling

### Using TPCH data

## pre-requisites

- Azure subscription
- Azure Storage Account
- Azure Synapse Analytics
- Load TPCH data
- I had to limit line items data as it was 150 Billions rows

## Dataset Rows

- Orders: 15,000,000,000
- Customers: 1,500,000,000
- Lineitems: 279,286,998

## Goal

- Use Dataflow to analyze data
- Use join and create year, month and day columns
- Do aggregation for year, month and day and calculate sum of numeric columns
- Try with Data flow first and then Synapse Spark
- Finally use Spark ML to do modelling with aggregated data

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe1.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe2.jpg "Architecture")

## Data Flow

- All the data are in parquet files
- Connect to the storage account and select the parquet files for Customers, orders and Lineitems
- Here is the end to end flow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe3.jpg "Architecture")

- Customers

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe4.jpg "Architecture")

- Orders

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe5.jpg "Architecture")

- Lineitems

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe6.jpg "Architecture")

- let's now join orders with line item

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe7.jpg "Architecture")

- now create year, month and day columns

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe8.jpg "Architecture")

- Now join with orders to get order details

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe9.jpg "Architecture")

- Now aggregate the data by year, month and day

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe10.jpg "Architecture")

- Calculate aggregates

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe11.jpg "Architecture")

- finally sink into ADLS as parquet

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe12.jpg "Architecture")

- now set the partition

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe13.jpg "Architecture")

## Next to Syanpse Spark Code

- Same above but PySpark
- Create a spark cluster with spark version 3.2
- Choose Extra large 5 nodes

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe14.jpg "Architecture")

- now time to code
- Let's bring libraries

```
from pyspark.sql.functions import *
```

- Load the data from ADLS
- Customer

```
%%pyspark
dfcustomer = spark.read.load('abfss://container@storageaccount.dfs.core.windows.net/tpch/CUSTOMER/*.parquet', format='parquet')
display(dfcustomer.limit(10))
```

- Create a View

```
dfcustomer.createOrReplaceTempView("customers")
```

- now orders

```
dforders = spark.read.load('abfss://containername@storageaccount.dfs.core.windows.net/tpch/ORDERS/*.parquet', format='parquet')
display(dforders.limit(10))
```

- Create columns for year, month, day

```
dforders = dforders.withColumn("year", year(col("O_ORDERDATE")))
dforders = dforders.withColumn("month", month(col("O_ORDERDATE")))
dforders = dforders.withColumn("day", dayofmonth(col("O_ORDERDATE")))
```

- Create a view

```
dforders.createOrReplaceTempView("orders")
```

- Line items load from ADLS

```
dflineitems = spark.read.load('abfss://containername@storageaccount.dfs.core.windows.net/tpch1/LINEITEM/*.parquet', format='parquet')
display(dflineitems.limit(10))
```

- Create view

```
dflineitems.createOrReplaceTempView("lineitems")`
```

- Now Aggregation

```
dfaggr = spark.sql("Select a.year, a.month, a.day, a.O_CUSTKEY, a.O_ORDERDATE, sum(a.O_TOTALPRICE) as O_TOTALPRICE , sum(b.L_DISCOUNT) as L_DISCOUNT, sum(b.L_QUANTITY) as L_QUANTITY, sum(b.L_TAX) as L_TAX, sum(b.L_LINENUMBER) as L_LINENUMBER, sum(b.L_EXTENDEDPRICE) as L_EXTENDEDPRICE from orders a join lineitems b on a.O_ORDERKEY = b.L_ORDERKEY join customers c on a.O_CUSTKEY = c.C_CUSTKEY group by a.year, a.month, a.day, a.O_CUSTKEY, a.O_ORDERDATE")
```

- Write the aggregation

```
dfaggr.repartition(6).write.mode("overwrite").parquet('abfss://root@synpasedlstore.dfs.core.windows.net/tpchoutputsparksql/')
```

- let's read and check

```
%%pyspark
df = spark.read.load('abfss://containername@storageaccount.dfs.core.windows.net/tpchoutputsparksql/*.snappy.parquet', format='parquet')
display(df.limit(10))
```

## Synapse Spark Training

- Read the data that was stored by synapse spark

```
%%pyspark
df = spark.read.load('abfss://containername@storageaccount.dfs.core.windows.net/tpchoutputsparksql/*.snappy.parquet', format='parquet')
display(df.limit(10))
```

- import libraries

```
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
```

- Display schema

```
df.printSchema()
```

- Featurization

```
featureassembler = VectorAssembler(inputCols = ["year","month", "day","L_DISCOUNT", "L_QUANTITY"], outputCol = "Independent Features")
```

- output

```
output = featureassembler.transform(df)
output.select("Independent Features").show()
```

- setup label

```
finalised_data = output.select("Independent Features", "O_TOTALPRICE")
```

- now split data

```
train_data, test_data = finalised_data.randomSplit([0.75, 0.25])
```

- setup regression

```
regressor = LinearRegression(featuresCol = 'Independent Features', labelCol = 'O_TOTALPRICE')
regressor = regressor.fit(train_data)
```

- display the metric

```
trainingSummary = regressor.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

```

- Now score or predict

```
pred_results = regressor.evaluate(test_data)
pred_results.predictions.show()
```

- Calculate results

```
lr_predictions = regressor.transform(test_data)
lr_predictions.select("prediction","O_TOTALPRICE","Independent Features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="O_TOTALPRICE",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

```

- Print score or prediction metrics

```
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
```

- Run the notebook

## Pipeline creation

- Create a new pipeline
- First bring data flow
- Runtime configuration
- 
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe15.jpg "Architecture")

- Now configure dataflow

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe16.jpg "Architecture")

- Next bring in synapse spark for Syanpse spark aggregation

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe17.jpg "Architecture")

- Next bring synapse spark for ML training

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe18.jpg "Architecture")

- Save the pipeline and publish
- Now run the end to end pipeline
- 
![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe19.jpg "Architecture")

- Now show the lineage

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/SyanpseSpark/images/tpchetoe20.jpg "Architecture")