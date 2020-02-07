# <font color='blue'>Week 4 - Machine Learning Intro</font>
Used:
Python and its libraries; NumPy and Pandas library.
Jupyter Notebook.


**Import the libraries**


```python
import pandas
import numpy
import seaborn
import matplotlib.pyplot as matPlotLibPyPlot
```

**Reading data from https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data in a dataframe called balance_data**


```python
balance_data = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)
```

**Print the dimensions of the dataset**


```python
'''
O/P:

Dataset Lenght::  625
Dataset Shape::  (625, 5)

'''
```




    '\nO/P:\n\nDataset Lenght::  625\nDataset Shape::  (625, 5)\n\n'




```python
print( "Dataset Length::" ,       balance_data.shape[0] ,
      "\nDataset Shape::" ,       balance_data.shape)

```

    Dataset Length:: 625 
    Dataset Shape:: (625, 5)
    


```python
balance_data.columns
```




    Int64Index([0, 1, 2, 3, 4], dtype='int64')




```python
balance_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 625 entries, 0 to 624
    Data columns (total 5 columns):
    0    625 non-null object
    1    625 non-null int64
    2    625 non-null int64
    3    625 non-null int64
    4    625 non-null int64
    dtypes: int64(4), object(1)
    memory usage: 24.5+ KB
    


```python
balance_data.describe()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>625.000000</td>
      <td>625.000000</td>
      <td>625.000000</td>
      <td>625.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.415346</td>
      <td>1.415346</td>
      <td>1.415346</td>
      <td>1.415346</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Show header of the dataset**


```python
balance_data.head(5)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
balance_data.sample(4)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>457</th>
      <td>L</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>255</th>
      <td>L</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>604</th>
      <td>L</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>238</th>
      <td>R</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
balance_data.keys()
```




    Int64Index([0, 1, 2, 3, 4], dtype='int64')




```python
balance_data.columns
```




    Int64Index([0, 1, 2, 3, 4], dtype='int64')



**Prepare X and Y**


```python
x = balance_data[[1, 2, 3, 4]] # here  x is independent variable choosen
y= balance_data[[0]]
```

**Print contents of X**


```python
x.values
```




    array([[1, 1, 1, 1],
           [1, 1, 1, 2],
           [1, 1, 1, 3],
           ...,
           [5, 5, 5, 3],
           [5, 5, 5, 4],
           [5, 5, 5, 5]], dtype=int64)




```python
'''# O/P must be

array([[1, 1, 1, 1],
       [1, 1, 1, 2],
       [1, 1, 1, 3],
       ...,
       [5, 5, 5, 3],
       [5, 5, 5, 4],
       [5, 5, 5, 5]], dtype=object)
       '''
```




    '# O/P must be\n\narray([[1, 1, 1, 1],\n       [1, 1, 1, 2],\n       [1, 1, 1, 3],\n       ...,\n       [5, 5, 5, 3],\n       [5, 5, 5, 4],\n       [5, 5, 5, 5]], dtype=object)\n       '



**Print contents of Y**


```python
y.values
```




    array([['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['L'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['L'],
           ['L'],
           ['L'],
           ['B'],
           ['R'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['L'],
           ['B']], dtype=object)




```python
'''This must be my O/P:

array(['B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L',
       'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L',
       'B', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R',
       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L',
       'B', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B',
       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L',
       'B', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R',
       'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'L', 'B', 'R', 'R', 'R',
       'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'B', 'R', 'L',
       'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R',
       'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'R', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
       'L', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'B', 'R', 'R', 'R', 'L',
       'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'B',
       'R', 'R', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
       'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
       'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'R', 'R', 'L',
       'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R',
       'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'R', 'L', 'L',
       'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'L',
       'L', 'B', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
       'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'B', 'R', 'L', 'L', 'R', 'R', 'R', 'L',
       'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'R', 'L', 'L',
       'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'R', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B', 'L', 'L', 'L', 'B', 'R', 'L', 'L', 'L', 'L', 'B', 'L', 'L',
       'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
       'B', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'L', 'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L',
       'B', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L',
       'B', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L', 'L',
       'B', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
       'B'], dtype=object)
       '''
```




    "This must be my O/P:\n\narray(['B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',\n       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L',\n       'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',\n       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L',\n       'B', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R',\n       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L',\n       'B', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B',\n       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L',\n       'B', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R',\n       'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'L', 'B', 'R', 'R', 'R',\n       'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',\n       'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'B', 'R', 'L',\n       'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R',\n       'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'R', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',\n       'L', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'B', 'R', 'R', 'R', 'L',\n       'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'B',\n       'R', 'R', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'R', 'R', 'R', 'R',\n       'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',\n       'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'R', 'R', 'L',\n       'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R',\n       'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'R', 'L', 'L',\n       'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'L',\n       'L', 'B', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',\n       'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'B', 'R', 'L', 'L', 'R', 'R', 'R', 'L',\n       'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'R', 'L', 'L',\n       'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'R', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B', 'L', 'L', 'L', 'B', 'R', 'L', 'L', 'L', 'L', 'B', 'L', 'L',\n       'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',\n       'B', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'L', 'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L',\n       'B', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L',\n       'B', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L', 'L',\n       'B', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',\n       'B'], dtype=object)\n       "



** Import the library which helps to split in train and test and split the data in training (70%) and testing (30%)**


```python
from sklearn.model_selection import train_test_split
```


```python
# myLinRegAlgo
x_train, x_test, y_train, y_test =  train_test_split(x,y,
                                                     test_size=0.30,
                                                     random_state =1029)
```


```python
from sklearn.linear_model import LinearRegression
'''we are not using Linear Regression ALGO herel we wanna learn now DECISION_TREE, 
so import DECISION_TREE'''
```




    'we are not using Linear Regression ALGO herel we wanna learn now DECISION_TREE, \nso import DECISION_TREE'



**import the DecisionTreeClassifier from sklearn.tree**


```python
from sklearn.tree import DecisionTreeClassifier
```

**Instantiate an object of DecisionTreeClassifier using entropy**


```python
# myDecTree = DecisionTreeClassifier()
myDecTree = DecisionTreeClassifier(criterion='entropy') #  criterion='gini'
```

**Fit DecisionTreeClassifier object to X_train and y_train**


```python
myDecTree.fit(x_train, y_train ) 
#so here I am adding trained data(by the train_split model, you can import any other models, 
#but here we are using train_test split named model) into my algorithm
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')



**Predict the labels for X_test**


```python
x_test
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>526</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>234</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>125</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>352</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>153</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>361</th>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>285</th>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>284</th>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>186</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>536</th>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>346</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>620</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>514</th>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>354</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>230</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>114</th>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>478</th>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>449</th>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>258</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>370</th>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>568</th>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>621</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>582</th>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>334</th>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>528</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>293</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>591</th>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>317</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>353</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>500</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>485</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>264</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>489</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>290</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>612</th>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>327</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>150</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>389</th>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>557</th>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>417</th>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 4 columns</p>
</div>




```python
x_test.values
```




    array([[5, 2, 1, 2],
           [2, 5, 2, 5],
           [1, 3, 2, 2],
           [2, 1, 1, 1],
           [2, 3, 4, 2],
           [1, 3, 3, 2],
           [3, 5, 1, 3],
           [2, 2, 1, 4],
           [3, 5, 3, 2],
           [3, 2, 3, 1],
           [3, 2, 2, 5],
           [1, 2, 5, 5],
           [2, 3, 3, 2],
           [1, 2, 1, 3],
           [5, 2, 3, 2],
           [2, 5, 3, 4],
           [3, 4, 5, 2],
           [2, 1, 4, 1],
           [5, 5, 5, 1],
           [5, 1, 3, 5],
           [2, 1, 2, 2],
           [3, 5, 1, 5],
           [2, 5, 2, 1],
           [1, 5, 3, 5],
           [2, 4, 1, 2],
           [4, 5, 1, 4],
           [1, 4, 1, 1],
           [4, 3, 5, 5],
           [3, 1, 2, 4],
           [1, 5, 1, 2],
           [2, 5, 3, 3],
           [2, 4, 4, 1],
           [3, 1, 1, 2],
           [1, 4, 3, 4],
           [4, 2, 5, 5],
           [3, 4, 3, 2],
           [3, 1, 3, 3],
           [4, 3, 4, 2],
           [5, 3, 2, 4],
           [2, 3, 4, 4],
           [1, 2, 2, 3],
           [4, 4, 1, 3],
           [4, 2, 4, 1],
           [4, 5, 3, 4],
           [4, 1, 3, 4],
           [4, 5, 2, 4],
           [4, 5, 1, 3],
           [1, 2, 1, 5],
           [5, 5, 2, 4],
           [5, 5, 4, 4],
           [4, 4, 3, 5],
           [1, 1, 2, 3],
           [1, 2, 4, 5],
           [4, 4, 1, 2],
           [1, 2, 2, 4],
           [5, 4, 5, 1],
           [1, 4, 4, 3],
           [5, 4, 5, 3],
           [5, 5, 2, 1],
           [4, 5, 5, 2],
           [1, 1, 4, 2],
           [1, 1, 3, 1],
           [3, 2, 5, 3],
           [2, 2, 5, 4],
           [3, 5, 3, 1],
           [1, 3, 1, 4],
           [5, 1, 3, 2],
           [4, 2, 2, 5],
           [5, 3, 3, 4],
           [3, 4, 3, 4],
           [5, 3, 3, 3],
           [3, 2, 2, 1],
           [5, 5, 3, 2],
           [5, 4, 5, 4],
           [2, 2, 3, 1],
           [1, 4, 3, 1],
           [4, 3, 4, 3],
           [1, 4, 5, 3],
           [2, 2, 2, 5],
           [2, 1, 1, 3],
           [2, 1, 5, 5],
           [3, 4, 5, 1],
           [4, 4, 4, 1],
           [4, 4, 3, 1],
           [1, 1, 2, 1],
           [4, 3, 4, 5],
           [4, 2, 5, 4],
           [2, 1, 3, 5],
           [1, 1, 1, 4],
           [1, 4, 1, 3],
           [4, 3, 3, 1],
           [4, 1, 3, 2],
           [3, 1, 4, 3],
           [4, 4, 2, 4],
           [3, 4, 1, 1],
           [2, 2, 5, 3],
           [5, 2, 2, 1],
           [2, 4, 2, 5],
           [3, 2, 2, 3],
           [2, 2, 1, 5],
           [1, 5, 4, 2],
           [5, 2, 3, 1],
           [4, 2, 2, 2],
           [5, 2, 2, 2],
           [4, 5, 4, 1],
           [4, 1, 3, 3],
           [1, 5, 2, 3],
           [2, 3, 1, 4],
           [5, 5, 4, 5],
           [2, 5, 2, 4],
           [2, 2, 2, 1],
           [3, 2, 3, 4],
           [1, 3, 5, 3],
           [4, 4, 1, 4],
           [5, 2, 2, 3],
           [3, 3, 4, 4],
           [4, 2, 4, 5],
           [5, 2, 4, 4],
           [2, 2, 4, 5],
           [1, 1, 3, 3],
           [1, 5, 1, 4],
           [4, 4, 2, 5],
           [4, 3, 1, 4],
           [1, 2, 4, 1],
           [3, 3, 3, 2],
           [5, 4, 1, 1],
           [1, 3, 2, 4],
           [5, 2, 1, 1],
           [3, 5, 3, 4],
           [1, 2, 2, 2],
           [4, 1, 4, 1],
           [5, 3, 1, 5],
           [5, 5, 1, 4],
           [1, 2, 5, 4],
           [4, 3, 1, 1],
           [2, 1, 2, 5],
           [3, 5, 5, 5],
           [5, 1, 4, 4],
           [3, 1, 4, 2],
           [5, 4, 1, 5],
           [4, 5, 1, 5],
           [3, 4, 2, 2],
           [2, 5, 1, 4],
           [5, 2, 4, 2],
           [2, 4, 4, 5],
           [4, 3, 5, 1],
           [1, 2, 4, 3],
           [1, 1, 2, 4],
           [1, 3, 1, 2],
           [4, 3, 5, 4],
           [2, 3, 4, 1],
           [5, 2, 5, 3],
           [3, 3, 3, 1],
           [5, 3, 4, 3],
           [5, 3, 4, 5],
           [3, 4, 5, 5],
           [3, 5, 4, 2],
           [4, 1, 3, 1],
           [3, 5, 5, 1],
           [4, 1, 5, 4],
           [5, 3, 4, 4],
           [5, 5, 5, 2],
           [5, 4, 2, 3],
           [1, 4, 4, 2],
           [3, 4, 2, 5],
           [5, 2, 1, 4],
           [3, 2, 4, 4],
           [5, 4, 4, 2],
           [3, 3, 4, 3],
           [3, 5, 1, 4],
           [1, 4, 4, 1],
           [5, 1, 1, 1],
           [2, 1, 3, 4],
           [4, 5, 3, 1],
           [1, 1, 4, 1],
           [3, 1, 3, 5],
           [1, 4, 5, 5],
           [4, 5, 3, 5],
           [3, 2, 4, 1],
           [5, 5, 3, 3],
           [3, 4, 1, 3],
           [1, 3, 4, 4],
           [1, 2, 1, 2],
           [2, 2, 1, 1],
           [4, 1, 3, 5],
           [5, 3, 2, 3],
           [1, 2, 5, 3],
           [4, 2, 4, 3]], dtype=int64)



**Import confusion_matrix from sklearn.metrics and show the confusion matrix**


```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#from sklearn.metrics import classification_report, confusion_matrix
```


```python
'''O/P must be:
[[ 0  8  5]
 [11 70  4]
 [ 8  5 77]]
'''
```




    'O/P must be:\n[[ 0  8  5]\n [11 70  4]\n [ 8  5 77]]\n'




```python
# Summary in short for last 2-3 steps:
# x_test data  == real time original raw data
# prediciton data == some forecast value I wanna predict e.g.tomorrow weather
# so x_test is original data
# so x_train is data feeded to the machine model to train the model 
# so x_train ==(food given too machine)
# now, how do we predict the forecasting is working good or not?
# how we can trust?
# Now 2 parameters given by nice mathematician & developers! 
# 1. find entropy or gini_entropy value, precision etc
# 2. use confision matrix to evaluate: how? follow this...
# 3. Confusion Matrix Use case:
#            find the diff between x_test (30% portion of x in full) result (#x_test)
#                   vs
#           confusion matrix result  (#myConfMatRes)
#    see and compare what the outcome and think!!

```


```python
x_used_prediction = myDecTree.predict(x_test)  # why used x_test?
# because x_test data is untouched data by machine and
# its a fresh data to really predict!
```


```python
myConfMatRes = confusion_matrix(y_test, x_used_prediction)
```


```python
myConfMatRes
```




    array([[ 1,  5,  4],
           [12, 80,  3],
           [12,  1, 70]], dtype=int64)




```python
x_test
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>526</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>234</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>125</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>352</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>153</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>361</th>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>285</th>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>284</th>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>186</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>536</th>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>346</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>620</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>514</th>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>131</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>354</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>230</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>114</th>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>478</th>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>449</th>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>258</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>370</th>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>568</th>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>621</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>582</th>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>334</th>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>528</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>293</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>591</th>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>317</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>353</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>500</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>485</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>264</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>489</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>290</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>612</th>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>327</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>150</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>389</th>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>557</th>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>417</th>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 4 columns</p>
</div>



**Show the decision tree for created for the data**


```python
import graphviz
from sklearn import tree
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-64-0c901118e340> in <module>()
    ----> 1 import graphviz
          2 from sklearn import tree
    

    ModuleNotFoundError: No module named 'graphviz'



```python
myDecTree_t1 = tree.export_graphviz(myDecTree, out_file=None)
myDecTree_pdf = graphviz.Source(myDecTree_t1)
myDecTree_pdf.render(filename= "myDecTreePdf", directory='output')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-65-6a08e6285dc8> in <module>()
    ----> 1 myDecTree_t1 = tree.export_graphviz(myDecTree, out_file=None)
          2 myDecTree_pdf = graphviz.Source(myDecTree_t1)
          3 myDecTree_pdf.render(filename= "myDecTreePdf", directory='output')
    

    NameError: name 'tree' is not defined



```python
#use this generated pdf to integrate here
```


```python
%matplotlib inline
```


```python
gui_pdf_myDecTree = tree.export_graphviz(myDecTree, 
                                         out_file=None, 
                                         filled=True,
                                         rounded=True,
                                         special_characters=True) 
graphviz.Source(gui_pdf_myDecTree)
```
