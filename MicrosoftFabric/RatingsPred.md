# Fabric Data Science predict overall ratings based on building maintenance records

## Use Case

- Using MaintenanceSurveyResults data set from kaggle
- Predict overall rating based on building maintenance records
- This is using Regression modelling
- using SparkML
- Here are the columns in dataset

```
['Datayear',
 'Country',
 'City',
 'Zip',
 'STATE',
 'Is_Delegated',
 'Response_Count',
 'Response_Count_By_Bldg',
 'Question_Desc',
 'Question_Category',
 '5_StarRating',
 '4_StarRating',
 '3_StarRating',
 '2_StarRating',
 '1_StarRating',
 '0_StarRtaing',
 'Overall_Satisfaction',
 'Q_QUARTILE',
 'FirstIssueOfConcern',
 'SecondIssueoFConcern',
 'ThrirdIssueOfConcern',
 'FourthIssueOfConcern',
 'FifthIssueOfConcern']
```

## Code

- Here is the code
- First load the data from Lake house
- Make sure load the data into Lake houe

```
df = spark.sql("SELECT * FROM DexLakeHouse.MaintenanceSurveyResults")
display(df)
```

- Count the records

```
df.count()
```

- output

```
139035
```

- Display statistics of the dataframe

```
display(df.describe())
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/spreg1.jpg "Architecture")

- import plot libraries

```
import seaborn as sns 
import matplotlib.pyplot as plt
```

- Display group by mean

```
display(df.groupby(['Question_Desc']).mean())
```

- print Covariance

```
df.cov('Response_Count', 'Response_Count_By_Bldg')
```

- now correleation

```
df.corr('Response_Count', 'Response_Count_By_Bldg')
```

- filter only columns to use
- Idea here is unnecessary columns will impact the model

```
feature_group = ['Datayear', 'Country', 'City', 'Zip', 'STATE', 'Question_Desc','Overall_Satisfaction']
data_counts = df.groupBy(feature_group).count().alias("counts")
```

- display count

```
display(data_counts)
```

![Architecture](https://github.com/balakreshnan/Samples2023/blob/main/MicrosoftFabric/Images/spreg2.jpg "Architecture")

- create a dataframe with only required columns

```
dfpred = df['Datayear', 'Country', 'City', 'Zip', 'STATE', 'Question_Desc','Overall_Satisfaction']
```

- now to the part of machine learning modelling
- First we need to convert all string column into categorical column

```
from pyspark.ml.feature import (VectorAssembler,VectorIndexer, OneHotEncoder,StringIndexer)
```

- convert all the columns

```
country_indexer = StringIndexer(inputCol='Country',outputCol='CountryIndex')
country_encoder = OneHotEncoder(inputCol='CountryIndex',outputCol='CountryVec')

city_indexer = StringIndexer(inputCol='City',outputCol='CityIndex')
city_encoder = OneHotEncoder(inputCol='CityIndex',outputCol='CityVec')

zip_indexer = StringIndexer(inputCol='Zip',outputCol='ZipIndex')
zip_encoder = OneHotEncoder(inputCol='ZipIndex',outputCol='ZipVec')

state_indexer = StringIndexer(inputCol='STATE',outputCol='StateIndex')
state_encoder = OneHotEncoder(inputCol='StateIndex',outputCol='StateVec')

question_indexer = StringIndexer(inputCol='Question_Desc',outputCol='QuestionIndex')
question_encoder = OneHotEncoder(inputCol='QuestionIndex',outputCol='QuestionVec')
```

- All the above columns will change to categorical columns
- Set the assemble for spark ML

```
assembler = VectorAssembler(inputCols=['Datayear',
 'CountryVec',
 'CityVec',
 'ZipVec',
 'StateVec',
 'QuestionVec'],outputCol='features')
```

- import all the necessary libraries

```
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
```

- Set the linear regression model
- Set the pipeline

```
from pyspark.ml import Pipeline
#from pyspark.ml.classification import LogisticRegression

#log_reg_survey = LogisticRegression(featuresCol='features',labelCol='Overall_Satisfaction')
regressor = LinearRegression(featuresCol = 'features', labelCol = 'Overall_Satisfaction')

pipeline = Pipeline(stages=[country_indexer,city_indexer, zip_indexer, state_indexer, question_indexer,
                           country_encoder,city_encoder, zip_encoder, state_encoder, question_encoder,
                           assembler,regressor])
```

- now time to split the data

```
data_train , data_test = dfpred.randomSplit([0.8,0.2], seed = 123)
```

- Now time to train the model

```
Model = pipeline.fit(data_train)
```

- now lets predict

```
pred = Model.transform(data_test)
```

- display the prediction

```
pred.select("prediction","Overall_Satisfaction","features").show(5)
```

- Calculate the RMSE

```
from pyspark.ml.evaluation import RegressionEvaluator
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="Overall_Satisfaction",
                                predictionCol="prediction",
                                metricName="rmse")

rmse = evaluator.evaluate(pred)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
```

- Calculate r2

```
y_true = pred.select("Overall_Satisfaction").toPandas()
y_pred = pred.select("prediction").toPandas()

import sklearn.metrics
r2_score = sklearn.metrics.r2_score(y_true, y_pred)
print('r2_score: {0}'.format(r2_score))
```