
# <font color='blue'>Week 11 - Machine Learning Intro</font>

### <font color='red'> Used: Python and its libraries; pySparks </font>
Notes:
1. 'label' 
2. 'FeatureVector' 


**Following references are used for this module**
* Dataset: https://archive.ics.uci.edu/ml/datasets/adult
* https://docs.databricks.com/applications/machine-learning/mllib/binary-classification-mllib-pipelines.html

**Installing spark library and setting the Java envoirement**


```python
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

    Collecting pyspark
    [?25l  Downloading https://files.pythonhosted.org/packages/9a/5a/271c416c1c2185b6cb0151b29a91fff6fcaed80173c8584ff6d20e46b465/pyspark-2.4.5.tar.gz (217.8MB)
    [K     |████████████████████████████████| 217.8MB 62kB/s 
    [?25hCollecting py4j==0.10.7
    [?25l  Downloading https://files.pythonhosted.org/packages/e3/53/c737818eb9a7dc32a7cd4f1396e787bd94200c3997c72c1dbe028587bd76/py4j-0.10.7-py2.py3-none-any.whl (197kB)
    [K     |████████████████████████████████| 204kB 46.5MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-2.4.5-py2.py3-none-any.whl size=218257927 sha256=c3b329e710d3b4d84bed826f85c13c9e7658dc6faccbb85bc97320e7ae65c3e7
      Stored in directory: /root/.cache/pip/wheels/bf/db/04/61d66a5939364e756eb1c1be4ec5bdce6e04047fc7929a3c3c
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.7 pyspark-2.4.5
    openjdk-8-jdk-headless is already the newest version (8u242-b08-0ubuntu3~18.04).
    0 upgraded, 0 newly installed, 0 to remove and 25 not upgraded.
    

**Import the necessary libraries and setting the spark session**


```python
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()
```

**Upload the BostonHousing.csv file**


```python
from google.colab import files
uploaded = files.upload()
```



     <input type="file" id="files-e376cca9-59f9-4fbd-8859-072cf80caad9" name="files[]" multiple disabled />
     <output id="result-e376cca9-59f9-4fbd-8859-072cf80caad9">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving adult.data to adult.data
    

# Load training data

**Read the adult.data file into a dataset and name the columns**



```python
dataset = spark.read.format("csv").option("header","false").load("adult.data").toDF('age',  
            'workclass',  'fnlwgt',  'education',  'education_num',  'marital_status',  'occupation',  'relationship',  
            'race',  'sex',  'capital_gain',  'capital_loss',  'hours_per_week',  'native_country',  'income')
```

**Print the schema of the dataset using describe and printschema functions**


```python
dataset.printSchema()
```

    root
     |-- age: string (nullable = true)
     |-- workclass: string (nullable = true)
     |-- fnlwgt: string (nullable = true)
     |-- education: string (nullable = true)
     |-- education_num: string (nullable = true)
     |-- marital_status: string (nullable = true)
     |-- occupation: string (nullable = true)
     |-- relationship: string (nullable = true)
     |-- race: string (nullable = true)
     |-- sex: string (nullable = true)
     |-- capital_gain: string (nullable = true)
     |-- capital_loss: string (nullable = true)
     |-- hours_per_week: string (nullable = true)
     |-- native_country: string (nullable = true)
     |-- income: string (nullable = true)
    
    

**Print first 5 records of the dataset**



```python
dataset.head(5)
```




    [Row(age='39', workclass=' State-gov', fnlwgt=' 77516', education=' Bachelors', education_num=' 13', marital_status=' Never-married', occupation=' Adm-clerical', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=' 2174', capital_loss=' 0', hours_per_week=' 40', native_country=' United-States', income=' <=50K'),
     Row(age='50', workclass=' Self-emp-not-inc', fnlwgt=' 83311', education=' Bachelors', education_num=' 13', marital_status=' Married-civ-spouse', occupation=' Exec-managerial', relationship=' Husband', race=' White', sex=' Male', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 13', native_country=' United-States', income=' <=50K'),
     Row(age='38', workclass=' Private', fnlwgt=' 215646', education=' HS-grad', education_num=' 9', marital_status=' Divorced', occupation=' Handlers-cleaners', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 40', native_country=' United-States', income=' <=50K'),
     Row(age='53', workclass=' Private', fnlwgt=' 234721', education=' 11th', education_num=' 7', marital_status=' Married-civ-spouse', occupation=' Handlers-cleaners', relationship=' Husband', race=' Black', sex=' Male', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 40', native_country=' United-States', income=' <=50K'),
     Row(age='28', workclass=' Private', fnlwgt=' 338409', education=' Bachelors', education_num=' 13', marital_status=' Married-civ-spouse', occupation=' Prof-specialty', relationship=' Wife', race=' Black', sex=' Female', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 40', native_country=' Cuba', income=' <=50K')]




```python
dataset.describe()
```




    DataFrame[summary: string, age: string, workclass: string, fnlwgt: string, education: string, education_num: string, marital_status: string, occupation: string, relationship: string, race: string, sex: string, capital_gain: string, capital_loss: string, hours_per_week: string, native_country: string, income: string]



## Changing Data Types (Dataset ----change_data_type----> Dataset)


```python
dataset = dataset.withColumn("age", dataset["age"].cast("Float"))
dataset = dataset.withColumn("fnlwgt", dataset["fnlwgt"].cast("Float"))
dataset = dataset.withColumn("education_num", dataset["education_num"].cast("Float"))
dataset = dataset.withColumn("capital_gain", dataset["capital_gain"].cast("Float"))
dataset = dataset.withColumn("capital_loss", dataset["capital_loss"].cast("Float"))
dataset = dataset.withColumn("hours_per_week", dataset["hours_per_week"].cast("Float"))
dataset.printSchema()

```

    root
     |-- age: float (nullable = true)
     |-- workclass: string (nullable = true)
     |-- fnlwgt: float (nullable = true)
     |-- education: string (nullable = true)
     |-- education_num: float (nullable = true)
     |-- marital_status: string (nullable = true)
     |-- occupation: string (nullable = true)
     |-- relationship: string (nullable = true)
     |-- race: string (nullable = true)
     |-- sex: string (nullable = true)
     |-- capital_gain: float (nullable = true)
     |-- capital_loss: float (nullable = true)
     |-- hours_per_week: float (nullable = true)
     |-- native_country: string (nullable = true)
     |-- income: string (nullable = true)
    
    


```python
dataset.head(5)
```




    [Row(age=39.0, workclass=' State-gov', fnlwgt=77516.0, education=' Bachelors', education_num=13.0, marital_status=' Never-married', occupation=' Adm-clerical', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=2174.0, capital_loss=0.0, hours_per_week=40.0, native_country=' United-States', income=' <=50K'),
     Row(age=50.0, workclass=' Self-emp-not-inc', fnlwgt=83311.0, education=' Bachelors', education_num=13.0, marital_status=' Married-civ-spouse', occupation=' Exec-managerial', relationship=' Husband', race=' White', sex=' Male', capital_gain=0.0, capital_loss=0.0, hours_per_week=13.0, native_country=' United-States', income=' <=50K'),
     Row(age=38.0, workclass=' Private', fnlwgt=215646.0, education=' HS-grad', education_num=9.0, marital_status=' Divorced', occupation=' Handlers-cleaners', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, native_country=' United-States', income=' <=50K'),
     Row(age=53.0, workclass=' Private', fnlwgt=234721.0, education=' 11th', education_num=7.0, marital_status=' Married-civ-spouse', occupation=' Handlers-cleaners', relationship=' Husband', race=' Black', sex=' Male', capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, native_country=' United-States', income=' <=50K'),
     Row(age=28.0, workclass=' Private', fnlwgt=338409.0, education=' Bachelors', education_num=13.0, marital_status=' Married-civ-spouse', occupation=' Prof-specialty', relationship=' Wife', race=' Black', sex=' Female', capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, native_country=' Cuba', income=' <=50K')]






```
# BEFORE...
[Row(age='39', workclass=' State-gov', fnlwgt=' 77516', education=' Bachelors', education_num=' 13', marital_status=' Never-married', occupation=' Adm-clerical', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=' 2174', capital_loss=' 0', hours_per_week=' 40', native_country=' United-States', income=' <=50K'),
 Row(age='50', workclass=' Self-emp-not-inc', fnlwgt=' 83311', education=' Bachelors', education_num=' 13', marital_status=' Married-civ-spouse', occupation=' Exec-managerial', relationship=' Husband', race=' White', sex=' Male', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 13', native_country=' United-States', income=' <=50K'),
 Row(age='38', workclass=' Private', fnlwgt=' 215646', education=' HS-grad', education_num=' 9', marital_status=' Divorced', occupation=' Handlers-cleaners', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 40', native_country=' United-States', income=' <=50K'),
 Row(age='53', workclass=' Private', fnlwgt=' 234721', education=' 11th', education_num=' 7', marital_status=' Married-civ-spouse', occupation=' Handlers-cleaners', relationship=' Husband', race=' Black', sex=' Male', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 40', native_country=' United-States', income=' <=50K'),
 Row(age='28', workclass=' Private', fnlwgt=' 338409', education=' Bachelors', education_num=' 13', marital_status=' Married-civ-spouse', occupation=' Prof-specialty', relationship=' Wife', race=' Black', sex=' Female', capital_gain=' 0', capital_loss=' 0', hours_per_week=' 40', native_country=' Cuba', income=' <=50K')]

# AFTER...
[Row(age=39.0, workclass=' State-gov', fnlwgt=77516.0, education=' Bachelors', education_num=13.0, marital_status=' Never-married', occupation=' Adm-clerical', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=2174.0, capital_loss=0.0, hours_per_week=40.0, native_country=' United-States', income=' <=50K'),
 Row(age=50.0, workclass=' Self-emp-not-inc', fnlwgt=83311.0, education=' Bachelors', education_num=13.0, marital_status=' Married-civ-spouse', occupation=' Exec-managerial', relationship=' Husband', race=' White', sex=' Male', capital_gain=0.0, capital_loss=0.0, hours_per_week=13.0, native_country=' United-States', income=' <=50K'),
 Row(age=38.0, workclass=' Private', fnlwgt=215646.0, education=' HS-grad', education_num=9.0, marital_status=' Divorced', occupation=' Handlers-cleaners', relationship=' Not-in-family', race=' White', sex=' Male', capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, native_country=' United-States', income=' <=50K'),
 Row(age=53.0, workclass=' Private', fnlwgt=234721.0, education=' 11th', education_num=7.0, marital_status=' Married-civ-spouse', occupation=' Handlers-cleaners', relationship=' Husband', race=' Black', sex=' Male', capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, native_country=' United-States', income=' <=50K'),
 Row(age=28.0, workclass=' Private', fnlwgt=338409.0, education=' Bachelors', education_num=13.0, marital_status=' Married-civ-spouse', occupation=' Prof-specialty', relationship=' Wife', race=' Black', sex=' Female', capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, native_country=' Cuba', income=' <=50K')]

```
 


```python
# integer columns are==

# dataset["age"].cast("Integer")
# dataset["fnlwgt"].cast("Integer")
# dataset["education_num"].cast("Integer")
# dataset["capital_gain"].cast("Integer")
# dataset["capital_loss"].cast("Integer")
# dataset["hours_per_week"].cast("Integer")

#                 +

# by deafult "income" one


dataset_isolated=dataset[['age','fnlwgt','education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'income']]
```


```python
dataset_isolated.head(5)
```




    [Row(age=39.0, fnlwgt=77516.0, education_num=13.0, capital_gain=2174.0, capital_loss=0.0, hours_per_week=40.0, income=' <=50K'),
     Row(age=50.0, fnlwgt=83311.0, education_num=13.0, capital_gain=0.0, capital_loss=0.0, hours_per_week=13.0, income=' <=50K'),
     Row(age=38.0, fnlwgt=215646.0, education_num=9.0, capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, income=' <=50K'),
     Row(age=53.0, fnlwgt=234721.0, education_num=7.0, capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, income=' <=50K'),
     Row(age=28.0, fnlwgt=338409.0, education_num=13.0, capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, income=' <=50K')]



### General Guidleines / Procedure:

1. Instantiate Logistic Regression object

       my_logi_reg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

2. Fit the model

        my_logi_reg_model = my_logi_reg.fit(dataset_isolated)

    "Fit the Model"--->
    * This will throw an error !!! because there is no 'features' named column in the dataset, so dataset.head(5)
    * IllegalArgumentException: 'Field "features" does not exist.\nAvailable fields: age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income'
    * We are missing |"featuresVector"|"label"| So lets do it. First to create that...

3. Print the coefficients and intercept for logistic regression

        print("Coefficients: " + str(my_logi_reg_model.coefficients))
        print("Intercept: " + str(my_logi_reg_model.intercept))

4. Explaining label and Features
   - **label** : 0 or 1
   - **feature** : has a vector of all the features that belong to that row

5. Instantiate an object of the Logistic Regression .....
  Create initial LogisticRegression model

        my_logi_reg1 = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

6. Splitting the data in training and testing to evaluate the model

        train_data,test_data = dataset.randomSplit([0.70,0.30])
        train_data.show(5)
        test_data.show(5)

7. Fitting our model to the training data 
        
        model = my_logi_reg1.fit(train_data)
        predictions = model.transform(test_data)

    Predictions:
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        Evaluate model
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
        evaluator.evaluate(predictions)
        
        result = model.evaluate(test_data)
        result.accuracy

    Showing the predictions against the features of the test data:

        predictions = model.transform(test_data.select('features'))
        predictions.show(5)


#### **So Its a clear that we need 2 missing colms to proceed further:**

1. 'label' <-----casting_str_indxr------income (str)

2. 'FeatureVector' <-----combines------ ['age','fnlwgt','education_num', 'capital_gain', 'capital_loss', 'hours_per_week', ]

**Make the predictions by considering the numeric column only**

Make Vector 1st...


```python
from pyspark.ml.feature import VectorAssembler

"""
    "VectorAssembler" has 2 usage cases:
      - Automatically identify categorical features.
      - Index all features, if all features are categorical
"""


from pyspark.ml.feature import VectorAssembler
my_vector_assembler = VectorAssembler(inputCols=['age','fnlwgt','education_num', 'capital_gain', 'capital_loss', 'hours_per_week'],
                                      outputCol='FeaturesVector',
                                      handleInvalid='error')

```


```python

my_vector_assembler.transform(dataset_isolated)
"""
--->

ERROR:
#IllegalArgumentException: 'Data type string of column income is not supported.'
"""
```




    "\n--->\n\nERROR:\n#IllegalArgumentException: 'Data type string of column income is not supported.'\n"




```python
""" We need to cast data type of column: "income" to overcome above error:
By making 'income' column -----> 'label' column
"""

from pyspark.ml.feature import StringIndexer
my_stringIndexr = StringIndexer(inputCol='income', 
                                outputCol='label' , 
                                handleInvalid='error')

dataset_isolated = my_stringIndexr.fit(dataset_isolated).transform(dataset_isolated)

dataset_isolated.head(3)
```




    [Row(age=39.0, fnlwgt=77516.0, education_num=13.0, capital_gain=2174.0, capital_loss=0.0, hours_per_week=40.0, income=' <=50K', label=0.0),
     Row(age=50.0, fnlwgt=83311.0, education_num=13.0, capital_gain=0.0, capital_loss=0.0, hours_per_week=13.0, income=' <=50K', label=0.0),
     Row(age=38.0, fnlwgt=215646.0, education_num=9.0, capital_gain=0.0, capital_loss=0.0, hours_per_week=40.0, income=' <=50K', label=0.0)]




```python
dataset_isolated.printSchema()
```

    root
     |-- age: float (nullable = true)
     |-- fnlwgt: float (nullable = true)
     |-- education_num: float (nullable = true)
     |-- capital_gain: float (nullable = true)
     |-- capital_loss: float (nullable = true)
     |-- hours_per_week: float (nullable = true)
     |-- income: string (nullable = true)
     |-- label: double (nullable = false)
    
    


```python
from pyspark.ml.feature import VectorAssembler
my_vector_assembler = VectorAssembler(inputCols=['age','fnlwgt','education_num', 'capital_gain', 'capital_loss', 'hours_per_week'],
                                      outputCol='featuresVector',
                                      handleInvalid='error')

temp_data = my_vector_assembler.transform(dataset_isolated)
```


```python
temp_data.head(4)
temp_data.show(4)
```

    +----+--------+-------------+------------+------------+--------------+------+-----+--------------------+
    | age|  fnlwgt|education_num|capital_gain|capital_loss|hours_per_week|income|label|      featuresVector|
    +----+--------+-------------+------------+------------+--------------+------+-----+--------------------+
    |39.0| 77516.0|         13.0|      2174.0|         0.0|          40.0| <=50K|  0.0|[39.0,77516.0,13....|
    |50.0| 83311.0|         13.0|         0.0|         0.0|          13.0| <=50K|  0.0|[50.0,83311.0,13....|
    |38.0|215646.0|          9.0|         0.0|         0.0|          40.0| <=50K|  0.0|[38.0,215646.0,9....|
    |53.0|234721.0|          7.0|         0.0|         0.0|          40.0| <=50K|  0.0|[53.0,234721.0,7....|
    +----+--------+-------------+------------+------------+--------------+------+-----+--------------------+
    only showing top 4 rows
    
    


```python
final_data = temp_data.select('featuresVector', 'label')
final_data.head(4)
final_data.show(4)
```

    +--------------------+-----+
    |      featuresVector|label|
    +--------------------+-----+
    |[39.0,77516.0,13....|  0.0|
    |[50.0,83311.0,13....|  0.0|
    |[38.0,215646.0,9....|  0.0|
    |[53.0,234721.0,7....|  0.0|
    +--------------------+-----+
    only showing top 4 rows
    
    

# Apply from step #1:


```python
my_logi_reg = LogisticRegression(featuresCol='featuresVector', labelCol='label', maxIter=10)

train_data, test_data = final_data.randomSplit([0.70, 0.30])

```


```python
my_logi_reg_model = my_logi_reg.fit(train_data)
```


```python
evalu = my_logi_reg_model.evaluate(final_data)
```


```python
evalu.accuracy
```




    0.8039679371026688




```python
evalu.featuresCol
```




    'featuresVector'




```python
evalu.labelCol
```




    'label'




```python
evalu.labels
```




    [0.0, 1.0]



# Prediction

Make the predictions by considering the numeric column only

# Apply from step #7:


```python
predicting = my_logi_reg_model.transform(test_data.select('featuresVector'))
```


```python
predicting.show(5)
```

    +--------------------+--------------------+--------------------+----------+
    |      featuresVector|       rawPrediction|         probability|prediction|
    +--------------------+--------------------+--------------------+----------+
    |[17.0,19752.0,7.0...|[3.03970495324610...|[0.95433597314859...|       0.0|
    |[17.0,27032.0,6.0...|[3.56117842679644...|[0.97237924531024...|       0.0|
    |[17.0,32763.0,6.0...|[3.49398482178582...|[0.97051613381038...|       0.0|
    |[17.0,34019.0,7.0...|[3.16660993902379...|[0.95955823385344...|       0.0|
    |[17.0,34088.0,8.0...|[2.83849173717566...|[0.94472074833253...|       0.0|
    +--------------------+--------------------+--------------------+----------+
    only showing top 5 rows
    
    


```python

```
