
# <font color='blue'>Week 10 - Machine Learning Intro</font>

### <font color='red'> Used: Python and its libraries; NumPy and Pandas library. Jupyter Notebook. </font>
### <font color='red'> Used: Tensorsflow 2.x and Sparks </font>


**Installing spark library and setting the Java envoirement**


```python
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

    Requirement already satisfied: pyspark in /usr/local/lib/python3.6/dist-packages (2.4.5)
    Requirement already satisfied: py4j==0.10.7 in /usr/local/lib/python3.6/dist-packages (from pyspark) (0.10.7)
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    openjdk-8-jdk-headless is already the newest version (8u242-b08-0ubuntu3~18.04).
    0 upgraded, 0 newly installed, 0 to remove and 25 not upgraded.
    

**Import the necessary libraries and setting the spark session**


```python
from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LinearRegressionExample")\
        .getOrCreate()

#or if __name__ == "__main__":
#    spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()
```

**Upload the BostonHousing.csv file**


```python
from google.colab import files
uploaded = files.upload()
```



     <input type="file" id="files-4a242169-15b3-4d97-b6c9-92bb174f640d" name="files[]" multiple disabled />
     <output id="result-4a242169-15b3-4d97-b6c9-92bb174f640d">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving BostonHousing.csv to BostonHousing.csv
    

**Read the csv file into a dataset**


```python
df_dataset_spark = spark.read.csv("BostonHousing.csv", header =True, inferSchema=True)
```

**Print the schema of the dataset using describe and printschema functions**


```python
df_dataset_spark.describe()
```




    DataFrame[summary: string, crim: string, zn: string, indus: string, chas: string, nox: string, rm: string, age: string, dis: string, rad: string, tax: string, ptratio: string, b: string, lstat: string, medv: string]




```python
df_dataset_spark.schema
```




    StructType(List(StructField(crim,DoubleType,true),StructField(zn,DoubleType,true),StructField(indus,DoubleType,true),StructField(chas,IntegerType,true),StructField(nox,DoubleType,true),StructField(rm,DoubleType,true),StructField(age,DoubleType,true),StructField(dis,DoubleType,true),StructField(rad,IntegerType,true),StructField(tax,IntegerType,true),StructField(ptratio,DoubleType,true),StructField(b,DoubleType,true),StructField(lstat,DoubleType,true),StructField(medv,DoubleType,true)))




```python
df_dataset_spark.printSchema()
```

    root
     |-- crim: double (nullable = true)
     |-- zn: double (nullable = true)
     |-- indus: double (nullable = true)
     |-- chas: integer (nullable = true)
     |-- nox: double (nullable = true)
     |-- rm: double (nullable = true)
     |-- age: double (nullable = true)
     |-- dis: double (nullable = true)
     |-- rad: integer (nullable = true)
     |-- tax: integer (nullable = true)
     |-- ptratio: double (nullable = true)
     |-- b: double (nullable = true)
     |-- lstat: double (nullable = true)
     |-- medv: double (nullable = true)
    
    


```python
df_dataset_spark.columns
```




    ['crim',
     'zn',
     'indus',
     'chas',
     'nox',
     'rm',
     'age',
     'dis',
     'rad',
     'tax',
     'ptratio',
     'b',
     'lstat',
     'medv']



<font color='blue'>**The prices of the house indicated by the variable 'MEDV' is our dependent variable. We want to predict the house prices.**</font>

**Import Features VectorAssembler and LinearRegression libraries**


```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
```

* **Put all featurs into one vector and name it as featuresVector using VectorAssembler**
* **Select "featuresVector" and target variable "medv" from transformed dataset**

Ref: https://spark.apache.org/docs/latest/ml-features#vectorassembler



```python
# Feeding all the features into 1 vector first
assembler = VectorAssembler(inputCols=[
                                    'crim',
                                    'zn',
                                    'indus',
                                    'chas',
                                    'nox',
                                    'rm',
                                    'age',
                                    'dis',
                                    'rad',
                                    'tax',
                                    'ptratio',
                                    'b',
                                    'lstat'],
                                 outputCol="featuresVector")

temp_output = assembler.transform(df_dataset_spark)
# print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
final_ouput_vectorAssembler = temp_output.select("featuresVector", "medv")
final_ouput_vectorAssembler.show(10, truncate=True)
```

    +--------------------+----+
    |      featuresVector|medv|
    +--------------------+----+
    |[0.00632,18.0,2.3...|24.0|
    |[0.02731,0.0,7.07...|21.6|
    |[0.02729,0.0,7.07...|34.7|
    |[0.03237,0.0,2.18...|33.4|
    |[0.06905,0.0,2.18...|36.2|
    |[0.02985,0.0,2.18...|28.7|
    |[0.08829,12.5,7.8...|22.9|
    |[0.14455,12.5,7.8...|27.1|
    |[0.21124,12.5,7.8...|16.5|
    |[0.17004,12.5,7.8...|18.9|
    +--------------------+----+
    only showing top 10 rows
    
    

**Split the data in training 70% and testing 30%**


```python
train_data, test_data = final_ouput_vectorAssembler.randomSplit([0.70,0.30])
```

##  After Vector part is done as above; NOW --> turn for Linear Regression part

**Instantiate an object of the linear regression class with featuresCol = 'featuresVector', labelCol = 'medv'**


```python
lin_reg_obj = LinearRegression(featuresCol='featuresVector', labelCol='medv')
```

**Fit the model/object to the training data**


```python
model_fit_lin_reg_obj= lin_reg_obj.fit(train_data) # test data is unused to save for future testing for model
```

**Predict the prices for test_data**

**call evaluate method of model**


```python
eval_01 = model_fit_lin_reg_obj.evaluate(test_data)
```


```python
eval_01.rootMeanSquaredError
```




    4.722295235248123




```python
eval_01.meanAbsoluteError
```




    3.5071201672368897




```python
eval_01.meanSquaredError
```




    22.30007228884713



Showing the predictions against the features of the test data

**Show the predicted house prices**


```python
prediction_01 = eval_01.predictions.show(10)
prediction_01
```

    +--------------------+----+------------------+
    |      featuresVector|medv|        prediction|
    +--------------------+----+------------------+
    |[0.00632,18.0,2.3...|24.0| 30.59938935369036|
    |[0.01311,90.0,1.2...|35.4|31.963964124139345|
    |[0.01439,60.0,2.9...|29.1|31.943353893506448|
    |[0.01501,90.0,1.2...|50.0| 44.01913860196597|
    |[0.01709,90.0,2.0...|30.1|27.315549465799243|
    |[0.01965,80.0,1.7...|20.1|19.879477504713556|
    |[0.02187,60.0,2.9...|31.1| 32.14437059892305|
    |[0.02731,0.0,7.07...|21.6| 25.87836507505765|
    |[0.02875,28.0,15....|25.0|29.508682320068687|
    |[0.03049,55.0,3.7...|31.2|28.541847799849464|
    +--------------------+----+------------------+
    only showing top 10 rows
    
    

**Print the coefficients and intercept of the regression model**


```python
coeffi = model_fit_lin_reg_obj.coefficients
interce = model_fit_lin_reg_obj.intercept

print(f"The co-efficients are: {coeffi}")

print(f"The intercept is: {interce}")
```

    The co-efficients are: [-0.04783635619245727,0.049711222746546005,0.0557739823676318,1.158210387521602,-20.22665069815588,3.540052001040454,0.02416624926471506,-1.3908317309861862,0.4016977656520664,-0.017266400953137695,-1.052248372865812,0.009190162750443365,-0.6624554852876057]
    The intercept is: 41.755020009600315
    

**Import the RegressionEvaluator library to show RMSE, MSE and MAE**


```python
from pyspark.ml.evaluation import RegressionEvaluator
```

**Print RMSE, MSE and MAE by providing the prediction and label columns**


```python
reg_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='medv', metricName='rmse')
```


```python
print(f'RMSE : {reg_evaluator.evaluate(eval_01.predictions)}')
```

    RMSE : 4.722295235248123
    


```python
print(f'MSE : {reg_evaluator.evaluate(eval_01.predictions, {reg_evaluator.metricName: "mse"})}')
```

    MSE : 22.30007228884713
    


```python
print(f'MAE : {reg_evaluator.evaluate(eval_01.predictions, {reg_evaluator.metricName: "mae"})}')
```

    MAE : 3.5071201672368897
    
