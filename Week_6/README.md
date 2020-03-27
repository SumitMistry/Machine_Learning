
# <font color='blue'>Week 6 - Machine Learning Intro</font>

### <font color='red'> Used: Python and its libraries; NumPy and Pandas library. Jupyter Notebook. </font>


**XGBoost Algorithm**
XGBoost is one of the most popular machine learning algorithm these days. Regardless of the type of prediction task at hand; regression or classification.


```python
%autosave 20
```



    Autosaving every 20 seconds
    

Reference links:

https://xgboost.readthedocs.io/en/latest/

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef

Import the Boston Housing dataset from scikit-learn and store it in a variable called boston_dataset.


```python
from sklearn.datasets import load_boston
#import sklearn.datasets.load_boston()
```


```python
boston_dataset = load_boston(return_X_y=False)
```


```python
boston_dataset
```




    {'data': array([[6.3200e-03, 1.8000e+01, 2.3100e+00, ..., 1.5300e+01, 3.9690e+02,
             4.9800e+00],
            [2.7310e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9690e+02,
             9.1400e+00],
            [2.7290e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9283e+02,
             4.0300e+00],
            ...,
            [6.0760e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
             5.6400e+00],
            [1.0959e-01, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9345e+02,
             6.4800e+00],
            [4.7410e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
             7.8800e+00]]),
     'target': array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15. ,
            18.9, 21.7, 20.4, 18.2, 19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6,
            15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21. , 12.7, 14.5, 13.2,
            13.1, 13.5, 18.9, 20. , 21. , 24.7, 30.8, 34.9, 26.6, 25.3, 24.7,
            21.2, 19.3, 20. , 16.6, 14.4, 19.4, 19.7, 20.5, 25. , 23.4, 18.9,
            35.4, 24.7, 31.6, 23.3, 19.6, 18.7, 16. , 22.2, 25. , 33. , 23.5,
            19.4, 22. , 17.4, 20.9, 24.2, 21.7, 22.8, 23.4, 24.1, 21.4, 20. ,
            20.8, 21.2, 20.3, 28. , 23.9, 24.8, 22.9, 23.9, 26.6, 22.5, 22.2,
            23.6, 28.7, 22.6, 22. , 22.9, 25. , 20.6, 28.4, 21.4, 38.7, 43.8,
            33.2, 27.5, 26.5, 18.6, 19.3, 20.1, 19.5, 19.5, 20.4, 19.8, 19.4,
            21.7, 22.8, 18.8, 18.7, 18.5, 18.3, 21.2, 19.2, 20.4, 19.3, 22. ,
            20.3, 20.5, 17.3, 18.8, 21.4, 15.7, 16.2, 18. , 14.3, 19.2, 19.6,
            23. , 18.4, 15.6, 18.1, 17.4, 17.1, 13.3, 17.8, 14. , 14.4, 13.4,
            15.6, 11.8, 13.8, 15.6, 14.6, 17.8, 15.4, 21.5, 19.6, 15.3, 19.4,
            17. , 15.6, 13.1, 41.3, 24.3, 23.3, 27. , 50. , 50. , 50. , 22.7,
            25. , 50. , 23.8, 23.8, 22.3, 17.4, 19.1, 23.1, 23.6, 22.6, 29.4,
            23.2, 24.6, 29.9, 37.2, 39.8, 36.2, 37.9, 32.5, 26.4, 29.6, 50. ,
            32. , 29.8, 34.9, 37. , 30.5, 36.4, 31.1, 29.1, 50. , 33.3, 30.3,
            34.6, 34.9, 32.9, 24.1, 42.3, 48.5, 50. , 22.6, 24.4, 22.5, 24.4,
            20. , 21.7, 19.3, 22.4, 28.1, 23.7, 25. , 23.3, 28.7, 21.5, 23. ,
            26.7, 21.7, 27.5, 30.1, 44.8, 50. , 37.6, 31.6, 46.7, 31.5, 24.3,
            31.7, 41.7, 48.3, 29. , 24. , 25.1, 31.5, 23.7, 23.3, 22. , 20.1,
            22.2, 23.7, 17.6, 18.5, 24.3, 20.5, 24.5, 26.2, 24.4, 24.8, 29.6,
            42.8, 21.9, 20.9, 44. , 50. , 36. , 30.1, 33.8, 43.1, 48.8, 31. ,
            36.5, 22.8, 30.7, 50. , 43.5, 20.7, 21.1, 25.2, 24.4, 35.2, 32.4,
            32. , 33.2, 33.1, 29.1, 35.1, 45.4, 35.4, 46. , 50. , 32.2, 22. ,
            20.1, 23.2, 22.3, 24.8, 28.5, 37.3, 27.9, 23.9, 21.7, 28.6, 27.1,
            20.3, 22.5, 29. , 24.8, 22. , 26.4, 33.1, 36.1, 28.4, 33.4, 28.2,
            22.8, 20.3, 16.1, 22.1, 19.4, 21.6, 23.8, 16.2, 17.8, 19.8, 23.1,
            21. , 23.8, 23.1, 20.4, 18.5, 25. , 24.6, 23. , 22.2, 19.3, 22.6,
            19.8, 17.1, 19.4, 22.2, 20.7, 21.1, 19.5, 18.5, 20.6, 19. , 18.7,
            32.7, 16.5, 23.9, 31.2, 17.5, 17.2, 23.1, 24.5, 26.6, 22.9, 24.1,
            18.6, 30.1, 18.2, 20.6, 17.8, 21.7, 22.7, 22.6, 25. , 19.9, 20.8,
            16.8, 21.9, 27.5, 21.9, 23.1, 50. , 50. , 50. , 50. , 50. , 13.8,
            13.8, 15. , 13.9, 13.3, 13.1, 10.2, 10.4, 10.9, 11.3, 12.3,  8.8,
             7.2, 10.5,  7.4, 10.2, 11.5, 15.1, 23.2,  9.7, 13.8, 12.7, 13.1,
            12.5,  8.5,  5. ,  6.3,  5.6,  7.2, 12.1,  8.3,  8.5,  5. , 11.9,
            27.9, 17.2, 27.5, 15. , 17.2, 17.9, 16.3,  7. ,  7.2,  7.5, 10.4,
             8.8,  8.4, 16.7, 14.2, 20.8, 13.4, 11.7,  8.3, 10.2, 10.9, 11. ,
             9.5, 14.5, 14.1, 16.1, 14.3, 11.7, 13.4,  9.6,  8.7,  8.4, 12.8,
            10.5, 17.1, 18.4, 15.4, 10.8, 11.8, 14.9, 12.6, 14.1, 13. , 13.4,
            15.2, 16.1, 17.8, 14.9, 14.1, 12.7, 13.5, 14.9, 20. , 16.4, 17.7,
            19.5, 20.2, 21.4, 19.9, 19. , 19.1, 19.1, 20.1, 19.9, 19.6, 23.2,
            29.8, 13.8, 13.3, 16.7, 12. , 14.6, 21.4, 23. , 23.7, 25. , 21.8,
            20.6, 21.2, 19.1, 20.6, 15.2,  7. ,  8.1, 13.6, 20.1, 21.8, 24.5,
            23.1, 19.7, 18.3, 21.2, 17.5, 16.8, 22.4, 20.6, 23.9, 22. , 11.9]),
     'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
            'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7'),
     'DESCR': ".. _boston_dataset:\n\nBoston house prices dataset\n---------------------------\n\n**Data Set Characteristics:**  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n.. topic:: References\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n",
     'filename': 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\boston_house_prices.csv'}



Print keys of boston dataset


```python
# o/p =dict_keys(['data', 'target', 'feature_names', 'DESCR'])
```


```python
boston_dataset.keys()
```




    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])




```python
boston_dataset.data
```




    array([[6.3200e-03, 1.8000e+01, 2.3100e+00, ..., 1.5300e+01, 3.9690e+02,
            4.9800e+00],
           [2.7310e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9690e+02,
            9.1400e+00],
           [2.7290e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9283e+02,
            4.0300e+00],
           ...,
           [6.0760e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
            5.6400e+00],
           [1.0959e-01, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9345e+02,
            6.4800e+00],
           [4.7410e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
            7.8800e+00]])




```python
boston_dataset.feature_names
```




    array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
           'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')




```python
len(boston_dataset)
```




    5




```python
print(boston_dataset.target)
```

    [24.  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9 15.  18.9 21.7 20.4
     18.2 19.9 23.1 17.5 20.2 18.2 13.6 19.6 15.2 14.5 15.6 13.9 16.6 14.8
     18.4 21.  12.7 14.5 13.2 13.1 13.5 18.9 20.  21.  24.7 30.8 34.9 26.6
     25.3 24.7 21.2 19.3 20.  16.6 14.4 19.4 19.7 20.5 25.  23.4 18.9 35.4
     24.7 31.6 23.3 19.6 18.7 16.  22.2 25.  33.  23.5 19.4 22.  17.4 20.9
     24.2 21.7 22.8 23.4 24.1 21.4 20.  20.8 21.2 20.3 28.  23.9 24.8 22.9
     23.9 26.6 22.5 22.2 23.6 28.7 22.6 22.  22.9 25.  20.6 28.4 21.4 38.7
     43.8 33.2 27.5 26.5 18.6 19.3 20.1 19.5 19.5 20.4 19.8 19.4 21.7 22.8
     18.8 18.7 18.5 18.3 21.2 19.2 20.4 19.3 22.  20.3 20.5 17.3 18.8 21.4
     15.7 16.2 18.  14.3 19.2 19.6 23.  18.4 15.6 18.1 17.4 17.1 13.3 17.8
     14.  14.4 13.4 15.6 11.8 13.8 15.6 14.6 17.8 15.4 21.5 19.6 15.3 19.4
     17.  15.6 13.1 41.3 24.3 23.3 27.  50.  50.  50.  22.7 25.  50.  23.8
     23.8 22.3 17.4 19.1 23.1 23.6 22.6 29.4 23.2 24.6 29.9 37.2 39.8 36.2
     37.9 32.5 26.4 29.6 50.  32.  29.8 34.9 37.  30.5 36.4 31.1 29.1 50.
     33.3 30.3 34.6 34.9 32.9 24.1 42.3 48.5 50.  22.6 24.4 22.5 24.4 20.
     21.7 19.3 22.4 28.1 23.7 25.  23.3 28.7 21.5 23.  26.7 21.7 27.5 30.1
     44.8 50.  37.6 31.6 46.7 31.5 24.3 31.7 41.7 48.3 29.  24.  25.1 31.5
     23.7 23.3 22.  20.1 22.2 23.7 17.6 18.5 24.3 20.5 24.5 26.2 24.4 24.8
     29.6 42.8 21.9 20.9 44.  50.  36.  30.1 33.8 43.1 48.8 31.  36.5 22.8
     30.7 50.  43.5 20.7 21.1 25.2 24.4 35.2 32.4 32.  33.2 33.1 29.1 35.1
     45.4 35.4 46.  50.  32.2 22.  20.1 23.2 22.3 24.8 28.5 37.3 27.9 23.9
     21.7 28.6 27.1 20.3 22.5 29.  24.8 22.  26.4 33.1 36.1 28.4 33.4 28.2
     22.8 20.3 16.1 22.1 19.4 21.6 23.8 16.2 17.8 19.8 23.1 21.  23.8 23.1
     20.4 18.5 25.  24.6 23.  22.2 19.3 22.6 19.8 17.1 19.4 22.2 20.7 21.1
     19.5 18.5 20.6 19.  18.7 32.7 16.5 23.9 31.2 17.5 17.2 23.1 24.5 26.6
     22.9 24.1 18.6 30.1 18.2 20.6 17.8 21.7 22.7 22.6 25.  19.9 20.8 16.8
     21.9 27.5 21.9 23.1 50.  50.  50.  50.  50.  13.8 13.8 15.  13.9 13.3
     13.1 10.2 10.4 10.9 11.3 12.3  8.8  7.2 10.5  7.4 10.2 11.5 15.1 23.2
      9.7 13.8 12.7 13.1 12.5  8.5  5.   6.3  5.6  7.2 12.1  8.3  8.5  5.
     11.9 27.9 17.2 27.5 15.  17.2 17.9 16.3  7.   7.2  7.5 10.4  8.8  8.4
     16.7 14.2 20.8 13.4 11.7  8.3 10.2 10.9 11.   9.5 14.5 14.1 16.1 14.3
     11.7 13.4  9.6  8.7  8.4 12.8 10.5 17.1 18.4 15.4 10.8 11.8 14.9 12.6
     14.1 13.  13.4 15.2 16.1 17.8 14.9 14.1 12.7 13.5 14.9 20.  16.4 17.7
     19.5 20.2 21.4 19.9 19.  19.1 19.1 20.1 19.9 19.6 23.2 29.8 13.8 13.3
     16.7 12.  14.6 21.4 23.  23.7 25.  21.8 20.6 21.2 19.1 20.6 15.2  7.
      8.1 13.6 20.1 21.8 24.5 23.1 19.7 18.3 21.2 17.5 16.8 22.4 20.6 23.9
     22.  11.9]
    


```python
print(boston_dataset.DESCR)
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    
    

# This is not a data frame!
## so we ned to create a df DataFrame NOW from DataSet.


```python
import pandas
```


```python
# this is wrong, it will have no data as .data is missing
pandas.DataFrame(data=boston_dataset, columns=boston_dataset.feature_names)
# this is wrong, it will have no data as .data is missing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_boston = pandas.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
df_boston
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>501</td>
      <td>0.06263</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.593</td>
      <td>69.1</td>
      <td>2.4786</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>391.99</td>
      <td>9.67</td>
    </tr>
    <tr>
      <td>502</td>
      <td>0.04527</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.120</td>
      <td>76.7</td>
      <td>2.2875</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>9.08</td>
    </tr>
    <tr>
      <td>503</td>
      <td>0.06076</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.976</td>
      <td>91.0</td>
      <td>2.1675</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>5.64</td>
    </tr>
    <tr>
      <td>504</td>
      <td>0.10959</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.794</td>
      <td>89.3</td>
      <td>2.3889</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>393.45</td>
      <td>6.48</td>
    </tr>
    <tr>
      <td>505</td>
      <td>0.04741</td>
      <td>0.0</td>
      <td>11.93</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>6.030</td>
      <td>80.8</td>
      <td>2.5050</td>
      <td>1.0</td>
      <td>273.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>7.88</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 13 columns</p>
</div>



Print dimensions of this dataset


```python
# o/p =(506, 13)
```


```python
df_boston.shape
```




    (506, 13)




```python
df_boston.describe
```




    <bound method NDFrame.describe of         CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
    0    0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
    1    0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
    2    0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
    3    0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
    4    0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   
    ..       ...   ...    ...   ...    ...    ...   ...     ...  ...    ...   
    501  0.06263   0.0  11.93   0.0  0.573  6.593  69.1  2.4786  1.0  273.0   
    502  0.04527   0.0  11.93   0.0  0.573  6.120  76.7  2.2875  1.0  273.0   
    503  0.06076   0.0  11.93   0.0  0.573  6.976  91.0  2.1675  1.0  273.0   
    504  0.10959   0.0  11.93   0.0  0.573  6.794  89.3  2.3889  1.0  273.0   
    505  0.04741   0.0  11.93   0.0  0.573  6.030  80.8  2.5050  1.0  273.0   
    
         PTRATIO       B  LSTAT  
    0       15.3  396.90   4.98  
    1       17.8  396.90   9.14  
    2       17.8  392.83   4.03  
    3       18.7  394.63   2.94  
    4       18.7  396.90   5.33  
    ..       ...     ...    ...  
    501     21.0  391.99   9.67  
    502     21.0  396.90   9.08  
    503     21.0  396.90   5.64  
    504     21.0  393.45   6.48  
    505     21.0  396.90   7.88  
    
    [506 rows x 13 columns]>



Print all features


```python
print(boston_dataset.feature_names)
```

    ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
     'B' 'LSTAT']
    


```python
df_boston.head(0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
print(df_boston.head(0))
```

    Empty DataFrame
    Columns: [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
    Index: []
    

Print description of the dataset

o/p =

"Boston House Prices dataset\n===========================\n\nNotes\n------\nData Set Characteristics:  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive\n    \n    :Median Value (attribute 14) is usually the target\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttp://archive.ics.uci.edu/ml/datasets/Housing\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n**References**\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)\n"


```python
boston_dataset.DESCR
```




    ".. _boston_dataset:\n\nBoston house prices dataset\n---------------------------\n\n**Data Set Characteristics:**  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n.. topic:: References\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n"



Create a DataFrame from boston.data


```python
df_boston = pandas.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
```


```python
df_boston.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
  </tbody>
</table>
</div>



Print 5 records from the dataset


```python
df_boston.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>430</td>
      <td>8.49213</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.584</td>
      <td>6.348</td>
      <td>86.1</td>
      <td>2.0527</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>83.45</td>
      <td>17.64</td>
    </tr>
    <tr>
      <td>53</td>
      <td>0.04981</td>
      <td>21.0</td>
      <td>5.64</td>
      <td>0.0</td>
      <td>0.439</td>
      <td>5.998</td>
      <td>21.4</td>
      <td>6.8147</td>
      <td>4.0</td>
      <td>243.0</td>
      <td>16.8</td>
      <td>396.90</td>
      <td>8.43</td>
    </tr>
    <tr>
      <td>469</td>
      <td>13.07510</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.580</td>
      <td>5.713</td>
      <td>56.7</td>
      <td>2.8237</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>14.76</td>
    </tr>
    <tr>
      <td>224</td>
      <td>0.31533</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.504</td>
      <td>8.266</td>
      <td>78.3</td>
      <td>2.8944</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>385.05</td>
      <td>4.14</td>
    </tr>
    <tr>
      <td>66</td>
      <td>0.04379</td>
      <td>80.0</td>
      <td>3.37</td>
      <td>0.0</td>
      <td>0.398</td>
      <td>5.787</td>
      <td>31.1</td>
      <td>6.6115</td>
      <td>4.0</td>
      <td>337.0</td>
      <td>16.1</td>
      <td>396.90</td>
      <td>10.24</td>
    </tr>
  </tbody>
</table>
</div>



Add boston.target to your pandas DataFrame with the name PRICE


```python
df_boston["CRIM"].sample(5) 
```




    489    0.18337
    54     0.01360
    471    4.03841
    55     0.01311
    93     0.02875
    Name: CRIM, dtype: float64




```python
#Here we created a new column into our dataset named : "boston_dataset"
df_boston["PRICE"] = boston_dataset.target 
# price is column name for dF dataFrame
# target is value getting from dataset going into this newly created column("price")
```


```python
#check its created or not
df_boston.sample(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>396</td>
      <td>5.87205</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.693</td>
      <td>6.405</td>
      <td>96.0</td>
      <td>1.6768</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.9</td>
      <td>19.37</td>
      <td>12.5</td>
    </tr>
    <tr>
      <td>234</td>
      <td>0.44791</td>
      <td>0.0</td>
      <td>6.2</td>
      <td>1.0</td>
      <td>0.507</td>
      <td>6.726</td>
      <td>66.5</td>
      <td>3.6519</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>360.2</td>
      <td>8.05</td>
      <td>29.0</td>
    </tr>
    <tr>
      <td>54</td>
      <td>0.01360</td>
      <td>75.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.410</td>
      <td>5.888</td>
      <td>47.6</td>
      <td>7.3197</td>
      <td>3.0</td>
      <td>469.0</td>
      <td>21.1</td>
      <td>396.9</td>
      <td>14.80</td>
      <td>18.9</td>
    </tr>
    <tr>
      <td>311</td>
      <td>0.79041</td>
      <td>0.0</td>
      <td>9.9</td>
      <td>0.0</td>
      <td>0.544</td>
      <td>6.122</td>
      <td>52.8</td>
      <td>2.6403</td>
      <td>4.0</td>
      <td>304.0</td>
      <td>18.4</td>
      <td>396.9</td>
      <td>5.98</td>
      <td>22.1</td>
    </tr>
  </tbody>
</table>
</div>



Call the info and describe methods for the dataframe data

o/p is =

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 506 entries, 0 to 505
Data columns (total 14 columns):
CRIM       506 non-null float64
ZN         506 non-null float64
INDUS      506 non-null float64
CHAS       506 non-null float64
NOX        506 non-null float64
RM         506 non-null float64
AGE        506 non-null float64
DIS        506 non-null float64
RAD        506 non-null float64
TAX        506 non-null float64
PTRATIO    506 non-null float64
B          506 non-null float64
LSTAT      506 non-null float64
PRICE      506 non-null float64
dtypes: float64(14)
memory usage: 55.4 KB



```python
df_boston.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    PRICE      506 non-null float64
    dtypes: float64(14)
    memory usage: 55.5 KB
    


```python
df_boston.describe
```




    <bound method NDFrame.describe of         CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
    0    0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
    1    0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
    2    0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
    3    0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
    4    0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   
    ..       ...   ...    ...   ...    ...    ...   ...     ...  ...    ...   
    501  0.06263   0.0  11.93   0.0  0.573  6.593  69.1  2.4786  1.0  273.0   
    502  0.04527   0.0  11.93   0.0  0.573  6.120  76.7  2.2875  1.0  273.0   
    503  0.06076   0.0  11.93   0.0  0.573  6.976  91.0  2.1675  1.0  273.0   
    504  0.10959   0.0  11.93   0.0  0.573  6.794  89.3  2.3889  1.0  273.0   
    505  0.04741   0.0  11.93   0.0  0.573  6.030  80.8  2.5050  1.0  273.0   
    
         PTRATIO       B  LSTAT  PRICE  
    0       15.3  396.90   4.98   24.0  
    1       17.8  396.90   9.14   21.6  
    2       17.8  392.83   4.03   34.7  
    3       18.7  394.63   2.94   33.4  
    4       18.7  396.90   5.33   36.2  
    ..       ...     ...    ...    ...  
    501     21.0  391.99   9.67   22.4  
    502     21.0  396.90   9.08   20.6  
    503     21.0  396.90   5.64   23.9  
    504     21.0  393.45   6.48   22.0  
    505     21.0  396.90   7.88   11.9  
    
    [506 rows x 14 columns]>



### Import the XGB Algorithm, mean_squared_error and numpy

bash
conda install py-xgboost


```python
import xgboost
from sklearn.metrics import mean_squared_error
import numpy
```


```python
df_boston.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
  </tbody>
</table>
</div>



Select your X and y from the data


```python
x = df_boston.iloc[:,:-1]
#y = df_boston.iloc[0:len(df_boston),-1:len(df_boston)]# this will give me the last only column
y = df_boston.iloc[:,-1]
```


```python
len(df_boston) #this is for rows 
```




    506




```python
len(df_boston.columns)
```




    14




```python
z = df_boston.iloc[0:len(df_boston),0:-1] #[R:C,R:C] #iloc= integer based location
#y = df_boston.iloc[0:len(df_boston),-1:len(df_boston)]# this will give me the last only column

z.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head(3)
```




    0    24.0
    1    21.6
    2    34.7
    Name: PRICE, dtype: float64



### Now you will convert the dataset into an optimized data structure called Dmatrix that XGBoost supports and gives it acclaimed performance and efficiency gains. You will use this later in the tutorial.


```python
# New cmd
data_dmatrix = xgboost.DMatrix(data=x,label=y)
```

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\core.py:588: FutureWarning: Series.base is deprecated and will be removed in a future version
      data.base is not None and isinstance(data, np.ndarray) \
    

### Split the data in 30% testing and 70% training 
#common formula since week-3, week-4.pdf


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=1029)
```

### Instantiate an XGBoost regressor object by calling the XGBRegressor() class from the XGBoost library with the hyper-parameters passed as arguments. For classification problems, you would have used the XGBClassifier() class.


```python
myXGBoost = xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
```

### Fit the regressor to the training set and make predictions on the test set using the familiar .fit() and .predict() methods.

o/p = [17:52:42] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.


```python
myXGBoost.fit(x_train, y_train)
```

    [21:39:23] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    

    C:\ProgramData\Anaconda3\lib\site-packages\xgboost\core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version
      if getattr(data, 'base', None) is not None and \
    




    XGBRegressor(alpha=10, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=0.3, gamma=0,
                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
                 max_depth=5, min_child_weight=1, missing=None, n_estimators=10,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                 silent=None, subsample=1, verbosity=1)



### Print the actual prices from test data


```python
myXGBoost.missing
```




    nan




```python
myXGBoost.learning_rate
```




    0.1




```python
myXGBoost.base_score
```




    0.5




```python
myXGBoost.kwargs
```




    {'alpha': 10}



# Print the actual prices from test data


```python
# Print the actual prices from test data
# x_test data was not used to train the model so this data is pure/untouched data
myXGBoost.predict(x_test)  # so x_test can give us a real time prediction.
```




    array([ 9.613199 , 10.932054 , 12.863521 , 14.545456 , 10.435616 ,
            9.868552 , 10.991049 ,  8.537203 , 10.1254425, 11.781072 ,
           22.830309 , 18.831198 , 13.259726 , 15.512948 , 18.93575  ,
           13.779034 , 16.963024 , 22.611881 , 13.0925255, 16.653694 ,
           15.592541 ,  8.304296 , 11.052871 , 14.394213 , 16.638666 ,
           12.728169 , 19.53332  , 14.984259 , 19.996819 , 17.17081  ,
            9.062304 , 14.937686 , 15.187487 , 18.831198 , 19.154423 ,
           12.384975 , 20.658506 ,  9.787031 ,  7.3919277, 15.046192 ,
           15.583125 , 18.435575 , 18.667007 , 16.061813 , 22.656502 ,
           15.475901 , 13.376813 , 10.999661 , 13.416661 , 14.954578 ,
           10.77917  , 10.749382 , 12.106483 , 11.696469 , 18.952229 ,
           11.25369  , 18.57501  , 16.054462 , 16.173088 ,  7.250405 ,
           12.489716 , 16.35532  ,  6.763396 , 17.010836 , 17.575794 ,
            8.790408 , 22.523949 , 15.4468155, 12.060374 , 23.778217 ,
           22.129562 , 18.18319  , 14.627773 , 13.259726 , 15.330975 ,
           13.330317 , 15.801362 , 11.415566 , 14.403312 , 15.025169 ,
           11.979152 , 20.327179 , 13.330317 ,  8.800793 , 23.106798 ,
           19.55888  , 13.156414 , 10.262989 , 21.725063 , 14.770722 ,
           13.801719 , 11.181586 , 19.747766 , 14.004316 , 15.821512 ,
           15.116333 , 13.53992  , 21.220598 , 16.272764 , 16.529219 ,
           16.280914 , 15.985626 , 14.857656 , 11.410328 , 11.9857645,
           13.639041 , 16.140192 , 16.208584 ,  8.552616 , 22.479782 ,
           13.53992  ,  9.438037 ,  7.4606333, 14.188744 , 13.376813 ,
            7.9746485, 10.220309 , 17.753311 , 11.028822 ,  7.6453543,
           11.420262 , 14.095941 , 12.75036  , 16.248932 , 14.59546  ,
           15.56374  , 23.16584  , 14.770722 , 10.563107 , 12.486195 ,
           11.515943 , 15.475901 , 19.567482 ,  9.638401 , 15.883735 ,
            8.889849 , 23.106798 , 18.611135 , 13.009741 , 16.146973 ,
           12.857576 , 15.349565 , 10.678354 , 16.132462 , 19.567755 ,
           14.364777 , 19.118233 , 15.3762245, 23.172472 , 23.16584  ,
           13.363259 , 12.486348 ], dtype=float32)



# Print the predicted prices

o/p=

array([15. , 26.6, 45.4, 20.8, 34.9, 21.9, 28.7,  7.2, 20. , 32.2, 24.1,
       18.5, 13.5, 27. , 23.1, 18.9, 24.5, 43.1, 19.8, 13.8, 15.6, 50. ,
       37.2, 46. , 50. , 21.2, 14.9, 19.6, 19.4, 18.6, 26.5, 32. , 10.9,
       20. , 21.4, 31. , 25. , 15.4, 13.1, 37.6, 37. , 18.9, 27.9, 50. ,
       14.4, 22. , 19.9, 21.6, 15.6, 15. , 32.4, 29.6, 20.4, 12.3, 19.1,
       14.9, 17.8,  8.8, 35.4, 11.5, 19.6, 20.6, 15.6, 19.9, 23.3, 22.3,
       24.8, 16.1, 22.8, 30.5, 20.4, 24.4, 16.6, 26.2, 16.4, 20.1, 13.9,
       19.4, 22.8, 13.8, 31.6, 10.5, 23.8, 22.4, 19.3, 22.2, 12.6, 19.4,
       22.2, 29.8,  9.6, 34.9, 21.4, 25.3, 32.9, 26.6, 14.6, 31.5, 23.3,
       33.3, 17.5, 19.1, 48.5, 17.1, 23.1, 28.4, 18.9, 13. , 17.2, 24.1,
       18.5, 21.8, 13.3, 23. , 14.1, 23.9, 24. , 17.2, 21.5, 19.1, 20.8,
       36. , 20.1,  8.7, 13.6, 22. , 22.2, 21.1, 13.4, 17.4, 20.1, 10.2,
       23.1, 10.2, 13.1, 14.3, 14.5,  7.2, 19.6, 20.6, 22.7, 26.4,  7.5,
       20.3, 50. ,  8.5, 20.3, 16.1, 22. , 19.6, 10.2, 23.2])


```python
#print(y_test.values.tolist())
y_test.values
```




    array([12.5, 14. , 18.8, 23.7, 10.8,  8.4, 15.6,  9.7, 13.5, 13.1, 50. ,
           23.6, 19. , 22.6, 24.8, 14.4, 22. , 38.7, 14.5, 19.4, 23.3, 16.3,
           14.1, 20.3, 21.9, 19.1, 37. , 29.8, 29. , 24.8, 11.8, 28.7, 19.8,
           33.2, 35.4, 14.8, 33.8, 10.2,  5.6, 21.6, 20.6, 31.5, 31.1, 23.3,
           50. , 25. , 24.5, 17. , 15. , 22.2, 15.4, 19.4, 13.2, 16.8, 32.4,
           15.6, 35.1, 27. , 23.3,  5. , 16.6, 25.2,  8.5, 24.8, 32.5,  8.4,
           50. , 22.9, 15.3, 43.5, 30.1, 28.4, 16.6, 18.5, 23.8, 19.3, 25.3,
           19.1, 27.1, 23.9, 27.5, 32. , 20.4, 13.1, 48.3, 31.5, 16.5, 14.9,
           33.3, 21.7, 21. , 18.4, 33.1, 19. , 23.6, 23.2, 23.1, 34.9, 28.6,
           27.9, 25. , 24.3, 23.4, 14.6, 22.7, 19.8, 36.2, 13.8,  8.7, 50. ,
           11.9, 13.4, 10.2, 23.1, 22. ,  8.8, 15.4, 27.5, 14.1,  7.2, 16.1,
           23. , 19.6, 20.9, 22.1, 24.6, 46. , 20.8, 20. , 25. , 19.5, 28.1,
           33.1,  6.3, 19.4,  7.5, 46.7, 24.1, 21.2, 24.6, 21.7, 18.2, 11.8,
           22. , 34.7, 19.7, 50. , 24.5, 36. , 45.4, 22.7, 20.2])



# Compare y's


```python
y.head(3)
```




    0    24.0
    1    21.6
    2    34.7
    Name: PRICE, dtype: float64




```python
y = df_boston.iloc[0:len(df_boston),-1:len(df_boston)]
y.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>21.6</td>
    </tr>
    <tr>
      <td>2</td>
      <td>34.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_test.head(3)
```




    396    12.5
    140    14.0
    124    18.8
    Name: PRICE, dtype: float64




```python
x,y, print("Hi")
```

    Hi
    




    (        CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
     0    0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
     1    0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
     2    0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
     3    0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
     4    0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   
     ..       ...   ...    ...   ...    ...    ...   ...     ...  ...    ...   
     501  0.06263   0.0  11.93   0.0  0.573  6.593  69.1  2.4786  1.0  273.0   
     502  0.04527   0.0  11.93   0.0  0.573  6.120  76.7  2.2875  1.0  273.0   
     503  0.06076   0.0  11.93   0.0  0.573  6.976  91.0  2.1675  1.0  273.0   
     504  0.10959   0.0  11.93   0.0  0.573  6.794  89.3  2.3889  1.0  273.0   
     505  0.04741   0.0  11.93   0.0  0.573  6.030  80.8  2.5050  1.0  273.0   
     
          PTRATIO       B  LSTAT  
     0       15.3  396.90   4.98  
     1       17.8  396.90   9.14  
     2       17.8  392.83   4.03  
     3       18.7  394.63   2.94  
     4       18.7  396.90   5.33  
     ..       ...     ...    ...  
     501     21.0  391.99   9.67  
     502     21.0  396.90   9.08  
     503     21.0  396.90   5.64  
     504     21.0  393.45   6.48  
     505     21.0  396.90   7.88  
     
     [506 rows x 13 columns],      PRICE
     0     24.0
     1     21.6
     2     34.7
     3     33.4
     4     36.2
     ..     ...
     501   22.4
     502   20.6
     503   23.9
     504   22.0
     505   11.9
     
     [506 rows x 1 columns], None)




```python
x_train, x_test, y_train, y_test
```




    (        CRIM    ZN  INDUS  CHAS    NOX     RM    AGE     DIS   RAD    TAX  \
     470  4.34879   0.0  18.10   0.0  0.580  6.167   84.0  3.0334  24.0  666.0   
     115  0.17134   0.0  10.01   0.0  0.547  5.928   88.2  2.4631   6.0  432.0   
     111  0.10084   0.0  10.01   0.0  0.547  6.715   81.6  2.6775   6.0  432.0   
     114  0.14231   0.0  10.01   0.0  0.547  6.254   84.2  2.2565   6.0  432.0   
     471  4.03841   0.0  18.10   0.0  0.532  6.229   90.7  3.0993  24.0  666.0   
     ..       ...   ...    ...   ...    ...    ...    ...     ...   ...    ...   
     142  3.32105   0.0  19.58   1.0  0.871  5.403  100.0  1.3216   5.0  403.0   
     371  9.23230   0.0  18.10   0.0  0.631  6.216  100.0  1.1691  24.0  666.0   
     192  0.08664  45.0   3.44   0.0  0.437  7.178   26.3  6.4798   5.0  398.0   
     253  0.36894  22.0   5.86   0.0  0.431  8.259    8.4  8.9067   7.0  330.0   
     138  0.24980   0.0  21.89   0.0  0.624  5.857   98.2  1.6686   4.0  437.0   
     
          PTRATIO       B  LSTAT  
     470     20.2  396.90  16.29  
     115     17.8  344.91  15.76  
     111     17.8  395.59  10.16  
     114     17.8  388.74  10.45  
     471     20.2  395.33  12.87  
     ..       ...     ...    ...  
     142     14.7  396.90  26.82  
     371     20.2  366.15   9.53  
     192     15.2  390.49   2.87  
     253     19.1  396.90   3.54  
     138     21.2  392.04  21.32  
     
     [354 rows x 13 columns],
              CRIM    ZN  INDUS  CHAS     NOX     RM    AGE     DIS   RAD    TAX  \
     396   5.87205   0.0  18.10   0.0  0.6930  6.405   96.0  1.6768  24.0  666.0   
     140   0.29090   0.0  21.89   0.0  0.6240  6.174   93.6  1.6119   4.0  437.0   
     124   0.09849   0.0  25.65   0.0  0.5810  5.879   95.8  2.0063   2.0  188.0   
     481   5.70818   0.0  18.10   0.0  0.5320  6.750   74.9  3.3317  24.0  666.0   
     444  12.80230   0.0  18.10   0.0  0.7400  5.854   96.6  1.8956  24.0  666.0   
     ..        ...   ...    ...   ...     ...    ...    ...     ...   ...    ...   
     248   0.16439  22.0   5.86   0.0  0.4310  6.433   49.1  7.8265   7.0  330.0   
     258   0.66351  20.0   3.97   0.0  0.6470  7.333  100.0  1.8946   5.0  264.0   
     280   0.03578  20.0   3.33   0.0  0.4429  7.820   64.5  4.6947   5.0  216.0   
     164   2.24236   0.0  19.58   0.0  0.6050  5.854   91.8  2.4220   5.0  403.0   
     463   5.82115   0.0  18.10   0.0  0.7130  6.513   89.9  2.8016  24.0  666.0   
     
          PTRATIO       B  LSTAT  
     396     20.2  396.90  19.37  
     140     21.2  388.08  24.16  
     124     19.1  379.38  17.58  
     481     20.2  393.07   7.74  
     444     20.2  240.52  23.79  
     ..       ...     ...    ...  
     248     19.1  374.71   9.52  
     258     13.0  383.29   7.79  
     280     14.9  387.31   3.76  
     164     14.7  395.11  11.64  
     463     20.2  393.82  10.29  
     
     [152 rows x 13 columns],
     470    19.9
     115    18.3
     111    22.8
     114    18.5
     471    19.6
            ... 
     142    13.4
     371    50.0
     192    36.4
     253    42.8
     138    13.3
     Name: PRICE, Length: 354, dtype: float64,
     396    12.5
     140    14.0
     124    18.8
     481    23.7
     444    10.8
            ... 
     248    24.5
     258    36.0
     280    45.4
     164    22.7
     463    20.2
     Name: PRICE, Length: 152, dtype: float64)




```python
myXGBoost.predict(x_test)
```




    array([ 9.613199 , 10.932054 , 12.863521 , 14.545456 , 10.435616 ,
            9.868552 , 10.991049 ,  8.537203 , 10.1254425, 11.781072 ,
           22.830309 , 18.831198 , 13.259726 , 15.512948 , 18.93575  ,
           13.779034 , 16.963024 , 22.611881 , 13.0925255, 16.653694 ,
           15.592541 ,  8.304296 , 11.052871 , 14.394213 , 16.638666 ,
           12.728169 , 19.53332  , 14.984259 , 19.996819 , 17.17081  ,
            9.062304 , 14.937686 , 15.187487 , 18.831198 , 19.154423 ,
           12.384975 , 20.658506 ,  9.787031 ,  7.3919277, 15.046192 ,
           15.583125 , 18.435575 , 18.667007 , 16.061813 , 22.656502 ,
           15.475901 , 13.376813 , 10.999661 , 13.416661 , 14.954578 ,
           10.77917  , 10.749382 , 12.106483 , 11.696469 , 18.952229 ,
           11.25369  , 18.57501  , 16.054462 , 16.173088 ,  7.250405 ,
           12.489716 , 16.35532  ,  6.763396 , 17.010836 , 17.575794 ,
            8.790408 , 22.523949 , 15.4468155, 12.060374 , 23.778217 ,
           22.129562 , 18.18319  , 14.627773 , 13.259726 , 15.330975 ,
           13.330317 , 15.801362 , 11.415566 , 14.403312 , 15.025169 ,
           11.979152 , 20.327179 , 13.330317 ,  8.800793 , 23.106798 ,
           19.55888  , 13.156414 , 10.262989 , 21.725063 , 14.770722 ,
           13.801719 , 11.181586 , 19.747766 , 14.004316 , 15.821512 ,
           15.116333 , 13.53992  , 21.220598 , 16.272764 , 16.529219 ,
           16.280914 , 15.985626 , 14.857656 , 11.410328 , 11.9857645,
           13.639041 , 16.140192 , 16.208584 ,  8.552616 , 22.479782 ,
           13.53992  ,  9.438037 ,  7.4606333, 14.188744 , 13.376813 ,
            7.9746485, 10.220309 , 17.753311 , 11.028822 ,  7.6453543,
           11.420262 , 14.095941 , 12.75036  , 16.248932 , 14.59546  ,
           15.56374  , 23.16584  , 14.770722 , 10.563107 , 12.486195 ,
           11.515943 , 15.475901 , 19.567482 ,  9.638401 , 15.883735 ,
            8.889849 , 23.106798 , 18.611135 , 13.009741 , 16.146973 ,
           12.857576 , 15.349565 , 10.678354 , 16.132462 , 19.567755 ,
           14.364777 , 19.118233 , 15.3762245, 23.172472 , 23.16584  ,
           13.363259 , 12.486348 ], dtype=float32)




```python
print(x.shape)
print(y.shape)
print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)
print(myXGBoost.predict(x_test).shape)
```

    (506, 13)
    (506, 1)
    (152, 13)
    (354, 13)
    (152,)
    (354,)
    (152,)
    

### Compute the RMSE by invoking the mean_sqaured_error function from sklearn's metrics module.

o/p = RMSE: 9.715257


```python
mse=numpy.sqrt(mean_squared_error(y_test, myXGBoost.predict(x_test)))
mse
```




    10.40452018557639




```python
import math
```


```python
rmse= "%f" % mse
rmse
```




    '10.404520'



# To be continue...

https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
