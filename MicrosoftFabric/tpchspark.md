# Process TPCH data in Spark

## Convert Parquet to Delta tables

- Read the Customer data
- Right click the files and customer get the full path

```
import pandas as pd
# Load data into pandas DataFrame from f"{mssparkutils.nbResPath}/builtin/titanic.csv"
dfcustomer = spark.read.parquet('abfss://xxxx@lakename.dfs.fabric.microsoft.com/xxxxxxxxxxxxxxxxxx/Files/CUSTOMER')
display(dfcustomer)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch12.jpg "Architecture")

- Do a count

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch13.jpg "Architecture")

- Convert to Delta table
- Print Schema

```
dfcustomer.printSchema
```

- Configure file size for delta

```
spark.conf.set("spark.sql.parquet.vorder.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")
```

- load the function needed to convert column data types

```
from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType, DecimalType, FloatType
```

- Convert columns which has 9 decimal values from 9 to 12 to accomodate big integer values

```
dfcustomer = dfcustomer.withColumn("C_ACCTBAL",col("C_ACCTBAL").cast(FloatType()))
dfcustomer = dfcustomer.withColumn("C_CUSTKEY",col("C_CUSTKEY").cast(DecimalType(12,0)))
```

- now write the data

```
table_name = "tpchcustomer"
dfcustomer.write.mode("overwrite").format("delta").save(f"abfss://xxxxxx@nelakename.dfs.fabric.microsoft.com/xxxxxx/Tables/{table_name}")
```

- now load line items

```
dflineitems = spark.read.parquet('abfss://xxxxxx@onelakename.dfs.fabric.microsoft.com/xxxxxxx/Files/LINEITEM')
display(dflineitems)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch14.jpg "Architecture")

- Convert columns which has 9 decimal values from 9 to 12 to accomodate big integer values

```
dflineitems = dflineitems.withColumn("L_PARTKEY",col("L_PARTKEY").cast(DecimalType(12,0)))
dflineitems = dflineitems.withColumn("L_SUPPKEY",col("L_SUPPKEY").cast(DecimalType(12,0)))
```

- now write the data

```
table_name = "tpchlineitem"
dflineitems.write.mode("overwrite").format("delta").save(f"abfss://xxxxx@onelakename.dfs.fabric.microsoft.com/xxxxxx/Tables/{table_name}")
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch11.jpg "Architecture")

- now process the orders


```
dforders = spark.read.parquet('abfss://xxxx@onelakename.dfs.fabric.microsoft.com/xxxxx/Files/ORDERS')
display(dforders)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/tpch14.jpg "Architecture")

- convert the data types

```
dforders = dforders.withColumn("O_CUSTKEY",col("O_CUSTKEY").cast(DecimalType(12,0)))
```

- Write the data back

```
table_name = "tpchorders"
dforders.write.mode("overwrite").format("delta").save(f"abfss://xxxx@onelakename.dfs.fabric.microsoft.com/xxxxxxx/Tables/{table_name}")
```
