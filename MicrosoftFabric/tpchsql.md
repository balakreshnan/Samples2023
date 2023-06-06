# Microsoft Fabric TPCH SQL

## Process TPCH data in Spark

### Convert Parquet to Delta tables

- tpchspark.md in this repo will cover the steps to convert parquet to delta tables
- [Spark Parquet to Delta](tpchspark.md)
- Read the Customer data
- Right click the files and customer get the full path

```
dfcustomer = spark.read.format('delta').load('abfss://xxxx@onelakename.dfs.fabric.microsoft.com/xxxxx/Tables/tpchcustomer')
```

- Read line items

```
dflineitem = spark.read.format('delta').load('abfss://xxxx@onelakename.dfs.fabric.microsoft.com/xxxx/Tables/tpchlineitem')
```

- Read orders

```
dforders = spark.read.format('delta').load('abfss://xxxx@onelakename.dfs.fabric.microsoft.com/xxxxxx/Tables/tpchorders')
```

- now join the orders and line item

```
dforders.join(dflineitem, dforders.O_ORDERKEY == dflineitem.L_ORDERKEY, "inner").show()
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch18.jpg "Architecture")

- above will process about 4.19 TB of data