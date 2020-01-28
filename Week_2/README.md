# <font color='blue'>Week 2 - Machine Learning Intro</font>
Used:
Python and its libraries; NumPy and Pandas library.
Jupyter Notebook.

#### 1. Import the numpy package and name it as np


```python
import numpy as np
```

**2. Create a vector/array from list [5, 5, 4, 6, 3]. Calculate mean, standard deviation and variance of the vector.**


```python
myVec = np.array([5,5,4,6,3])
```


```python
print(myVec)
```

    [5 5 4 6 3]



```python
np.mean(myVec)
```




    4.6




```python
np.std(myVec)
```




    1.019803902718557




```python
np.var(myVec)
```




    1.04



**3. Find indices of non-zero elements from [1,2,0,0,4,0]**


```python
test_01 = np.array([1,2,0,0,4,0])
print(test_01)
```

    [1 2 0 0 4 0]



```python
np.nonzero(test_01)
```




    (array([0, 1, 4]),)



**4. Create an array arr of length 15, filled with 1,2,3,...,15**


```python
arr = np.linspace(1,15,15)
print(arr)
```

    [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]



```python
arr = np.arange(1,16)
print(arr)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]


**5. Change the shape of arr to 3 rows and 5 columns**


```python
arr.shape =(3,5)
print(arr)
```

    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]]


**6. Change the shape of arr to 5 rows and 3 columns**


```python
arr.shape =(5,3)
print(arr)
```

    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]
     [13 14 15]]


# Important note for Sumit- practice


**Boolean array indexing:** Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition.

**7. Select elements from arr that are greater than 8.**


```python
# o/p = array([ 9, 10, 11, 12, 13, 14, 15])
```


```python
import numpy
arr_8plus = numpy.array(arr >8)
print(arr_8plus)
```

    [[False False False]
     [False False False]
     [False False  True]
     [ True  True  True]
     [ True  True  True]]



```python
arr_8plus_idx = (arr > 8)
print(arr_8plus_idx)
```

    [[False False False]
     [False False False]
     [False False  True]
     [ True  True  True]
     [ True  True  True]]



```python
arr[arr_8plus_idx]

```




    array([ 9, 10, 11, 12, 13, 14, 15])



# Pandas
### Step 1. Import the pandas library and name it as pd


```python
import pandas as pd
```

### Step 2. Import the dataset from `students.csv` and name it as students


```python
student_csv= pd.read_csv('students.csv')
```

### Step 3. Show the first 25 entries


```python
student_csv.head(25)
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
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GP</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>2</td>
      <td>health</td>
      <td>services</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>15</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>3</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>4</td>
      <td>3</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>2</td>
      <td>2</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>other</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>3</td>
      <td>2</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>4</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>14</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>teacher</td>
      <td>health</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>2</td>
      <td>1</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>10</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>services</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>3</td>
      <td>teacher</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>2</td>
      <td>2</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>14</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>services</td>
      <td>services</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>13</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>3</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GP</td>
      <td>M</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>2</td>
      <td>services</td>
      <td>services</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>16</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>4</td>
      <td>3</td>
      <td>health</td>
      <td>other</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>8</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>3</td>
      <td>teacher</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>12</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>22</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>4</td>
      <td>2</td>
      <td>teacher</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>15</td>
      <td>15</td>
      <td>16</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>2</td>
      <td>2</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>13</td>
      <td>13</td>
      <td>12</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>2</td>
      <td>4</td>
      <td>services</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>10</td>
      <td>9</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 33 columns</p>
</div>



### Step 4. Show the last 10 entries


```python
student_csv.tail(10)
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
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>385</th>
      <td>MS</td>
      <td>F</td>
      <td>18</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>2</td>
      <td>2</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>386</th>
      <td>MS</td>
      <td>F</td>
      <td>18</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>teacher</td>
      <td>at_home</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>387</th>
      <td>MS</td>
      <td>F</td>
      <td>19</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>2</td>
      <td>3</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>MS</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>1</td>
      <td>teacher</td>
      <td>services</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>9</td>
      <td>8</td>
    </tr>
    <tr>
      <th>389</th>
      <td>MS</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>MS</td>
      <td>M</td>
      <td>20</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>2</td>
      <td>2</td>
      <td>services</td>
      <td>services</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>11</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>391</th>
      <td>MS</td>
      <td>M</td>
      <td>17</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>1</td>
      <td>services</td>
      <td>services</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>14</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>392</th>
      <td>MS</td>
      <td>M</td>
      <td>21</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>393</th>
      <td>MS</td>
      <td>M</td>
      <td>18</td>
      <td>R</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>2</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>11</td>
      <td>12</td>
      <td>10</td>
    </tr>
    <tr>
      <th>394</th>
      <td>MS</td>
      <td>M</td>
      <td>19</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>other</td>
      <td>at_home</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>9</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>



### Step 5. Print concise summary of the dataset using the info() method


```python
student_csv.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 395 entries, 0 to 394
    Data columns (total 33 columns):
    school        395 non-null object
    sex           395 non-null object
    age           395 non-null int64
    address       395 non-null object
    famsize       395 non-null object
    Pstatus       395 non-null object
    Medu          395 non-null int64
    Fedu          395 non-null int64
    Mjob          395 non-null object
    Fjob          395 non-null object
    reason        395 non-null object
    guardian      395 non-null object
    traveltime    395 non-null int64
    studytime     395 non-null int64
    failures      395 non-null int64
    schoolsup     395 non-null object
    famsup        395 non-null object
    paid          395 non-null object
    activities    395 non-null object
    nursery       395 non-null object
    higher        395 non-null object
    internet      395 non-null object
    romantic      395 non-null object
    famrel        395 non-null int64
    freetime      395 non-null int64
    goout         395 non-null int64
    Dalc          395 non-null int64
    Walc          395 non-null int64
    health        395 non-null int64
    absences      395 non-null int64
    G1            395 non-null int64
    G2            395 non-null int64
    G3            395 non-null int64
    dtypes: int64(16), object(17)
    memory usage: 101.9+ KB


### Step 6. Find number of observations in the dataset


```python
student_csv.shape[0]
```




    395



### Step 7. Find number of columns in the dataset


```python
student_csv.shape[1]
```




    33



### Step 8. Print all columns names of the dataset.


```python
student_csv.head(2)
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
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GP</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 33 columns</p>
</div>




```python
student_csv.loc[1:0]
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
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 33 columns</p>
</div>




```python
student_csv.columns
```




    Index(['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
           'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
           'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
           'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
           'Walc', 'health', 'absences', 'G1', 'G2', 'G3'],
          dtype='object')



### Step 9. How is the dataset indexed? 
OR 
### Print the indices of the dataset.


```python
# "the index" (aka "the labels")
# O/P = RangeIndex(start=0, stop=395, step=1)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394]

```




    [0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28,
     29,
     30,
     31,
     32,
     33,
     34,
     35,
     36,
     37,
     38,
     39,
     40,
     41,
     42,
     43,
     44,
     45,
     46,
     47,
     48,
     49,
     50,
     51,
     52,
     53,
     54,
     55,
     56,
     57,
     58,
     59,
     60,
     61,
     62,
     63,
     64,
     65,
     66,
     67,
     68,
     69,
     70,
     71,
     72,
     73,
     74,
     75,
     76,
     77,
     78,
     79,
     80,
     81,
     82,
     83,
     84,
     85,
     86,
     87,
     88,
     89,
     90,
     91,
     92,
     93,
     94,
     95,
     96,
     97,
     98,
     99,
     100,
     101,
     102,
     103,
     104,
     105,
     106,
     107,
     108,
     109,
     110,
     111,
     112,
     113,
     114,
     115,
     116,
     117,
     118,
     119,
     120,
     121,
     122,
     123,
     124,
     125,
     126,
     127,
     128,
     129,
     130,
     131,
     132,
     133,
     134,
     135,
     136,
     137,
     138,
     139,
     140,
     141,
     142,
     143,
     144,
     145,
     146,
     147,
     148,
     149,
     150,
     151,
     152,
     153,
     154,
     155,
     156,
     157,
     158,
     159,
     160,
     161,
     162,
     163,
     164,
     165,
     166,
     167,
     168,
     169,
     170,
     171,
     172,
     173,
     174,
     175,
     176,
     177,
     178,
     179,
     180,
     181,
     182,
     183,
     184,
     185,
     186,
     187,
     188,
     189,
     190,
     191,
     192,
     193,
     194,
     195,
     196,
     197,
     198,
     199,
     200,
     201,
     202,
     203,
     204,
     205,
     206,
     207,
     208,
     209,
     210,
     211,
     212,
     213,
     214,
     215,
     216,
     217,
     218,
     219,
     220,
     221,
     222,
     223,
     224,
     225,
     226,
     227,
     228,
     229,
     230,
     231,
     232,
     233,
     234,
     235,
     236,
     237,
     238,
     239,
     240,
     241,
     242,
     243,
     244,
     245,
     246,
     247,
     248,
     249,
     250,
     251,
     252,
     253,
     254,
     255,
     256,
     257,
     258,
     259,
     260,
     261,
     262,
     263,
     264,
     265,
     266,
     267,
     268,
     269,
     270,
     271,
     272,
     273,
     274,
     275,
     276,
     277,
     278,
     279,
     280,
     281,
     282,
     283,
     284,
     285,
     286,
     287,
     288,
     289,
     290,
     291,
     292,
     293,
     294,
     295,
     296,
     297,
     298,
     299,
     300,
     301,
     302,
     303,
     304,
     305,
     306,
     307,
     308,
     309,
     310,
     311,
     312,
     313,
     314,
     315,
     316,
     317,
     318,
     319,
     320,
     321,
     322,
     323,
     324,
     325,
     326,
     327,
     328,
     329,
     330,
     331,
     332,
     333,
     334,
     335,
     336,
     337,
     338,
     339,
     340,
     341,
     342,
     343,
     344,
     345,
     346,
     347,
     348,
     349,
     350,
     351,
     352,
     353,
     354,
     355,
     356,
     357,
     358,
     359,
     360,
     361,
     362,
     363,
     364,
     365,
     366,
     367,
     368,
     369,
     370,
     371,
     372,
     373,
     374,
     375,
     376,
     377,
     378,
     379,
     380,
     381,
     382,
     383,
     384,
     385,
     386,
     387,
     388,
     389,
     390,
     391,
     392,
     393,
     394]




```python
idx0 = student_csv.index
idx1 = list(student_csv.index)
print(idx0, idx1)
```

    RangeIndex(start=0, stop=395, step=1) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394]


### Step 10. What is the data type of each column?


```python
student_csv.info() #schema
student_csv.dtypes #data type
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 395 entries, 0 to 394
    Data columns (total 33 columns):
    school        395 non-null object
    sex           395 non-null object
    age           395 non-null int64
    address       395 non-null object
    famsize       395 non-null object
    Pstatus       395 non-null object
    Medu          395 non-null int64
    Fedu          395 non-null int64
    Mjob          395 non-null object
    Fjob          395 non-null object
    reason        395 non-null object
    guardian      395 non-null object
    traveltime    395 non-null int64
    studytime     395 non-null int64
    failures      395 non-null int64
    schoolsup     395 non-null object
    famsup        395 non-null object
    paid          395 non-null object
    activities    395 non-null object
    nursery       395 non-null object
    higher        395 non-null object
    internet      395 non-null object
    romantic      395 non-null object
    famrel        395 non-null int64
    freetime      395 non-null int64
    goout         395 non-null int64
    Dalc          395 non-null int64
    Walc          395 non-null int64
    health        395 non-null int64
    absences      395 non-null int64
    G1            395 non-null int64
    G2            395 non-null int64
    G3            395 non-null int64
    dtypes: int64(16), object(17)
    memory usage: 101.9+ KB





    school        object
    sex           object
    age            int64
    address       object
    famsize       object
    Pstatus       object
    Medu           int64
    Fedu           int64
    Mjob          object
    Fjob          object
    reason        object
    guardian      object
    traveltime     int64
    studytime      int64
    failures       int64
    schoolsup     object
    famsup        object
    paid          object
    activities    object
    nursery       object
    higher        object
    internet      object
    romantic      object
    famrel         int64
    freetime       int64
    goout          int64
    Dalc           int64
    Walc           int64
    health         int64
    absences       int64
    G1             int64
    G2             int64
    G3             int64
    dtype: object



### Step 11. Print only the `Fjob` column


```python
print(student_csv.Fjob[:])
```

    0       teacher
    1         other
    2         other
    3      services
    4         other
    5         other
    6         other
    7       teacher
    8         other
    9         other
    10       health
    11        other
    12     services
    13        other
    14        other
    15        other
    16     services
    17        other
    18     services
    19        other
    20        other
    21       health
    22        other
    23        other
    24       health
    25     services
    26        other
    27     services
    28        other
    29      teacher
             ...   
    365       other
    366    services
    367    services
    368    services
    369     teacher
    370    services
    371    services
    372     at_home
    373       other
    374       other
    375       other
    376       other
    377    services
    378       other
    379       other
    380     teacher
    381       other
    382    services
    383    services
    384       other
    385       other
    386     at_home
    387       other
    388    services
    389       other
    390    services
    391    services
    392       other
    393       other
    394     at_home
    Name: Fjob, Length: 395, dtype: object



```python
print(student_csv['Fjob'])
```

    0       teacher
    1         other
    2         other
    3      services
    4         other
    5         other
    6         other
    7       teacher
    8         other
    9         other
    10       health
    11        other
    12     services
    13        other
    14        other
    15        other
    16     services
    17        other
    18     services
    19        other
    20        other
    21       health
    22        other
    23        other
    24       health
    25     services
    26        other
    27     services
    28        other
    29      teacher
             ...   
    365       other
    366    services
    367    services
    368    services
    369     teacher
    370    services
    371    services
    372     at_home
    373       other
    374       other
    375       other
    376       other
    377    services
    378       other
    379       other
    380     teacher
    381       other
    382    services
    383    services
    384       other
    385       other
    386     at_home
    387       other
    388    services
    389       other
    390    services
    391    services
    392       other
    393       other
    394     at_home
    Name: Fjob, Length: 395, dtype: object


### Step 12. How many different occupations are there for `Fjob` in this dataset?


```python
student_csv.Fjob.unique()
```




    array(['teacher', 'other', 'services', 'health', 'at_home'], dtype=object)




```python
len(student_csv.Fjob.unique())
```




    5



### Step 13. What is the most frequent occupation for `Fjob`?


```python
temp1 = student_csv.Fjob.describe().tail(2)
temp2 = temp1.head(1)
print(temp2)
```

    top    other
    Name: Fjob, dtype: object



```python
student_csv.Fjob.value_counts()
```




    other       217
    services    111
    teacher      29
    at_home      20
    health       18
    Name: Fjob, dtype: int64



### Step 14. Generate descriptive statistics that summarize the central tendency, dispersion and shape of the dataset's distribution, excluding ``NaN`` values (Summarize the DataFrame using the method describe()). 


```python
student_csv.describe() ##printing summary of the numeric column only
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
      <th>age</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>traveltime</th>
      <th>studytime</th>
      <th>failures</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.696203</td>
      <td>2.749367</td>
      <td>2.521519</td>
      <td>1.448101</td>
      <td>2.035443</td>
      <td>0.334177</td>
      <td>3.944304</td>
      <td>3.235443</td>
      <td>3.108861</td>
      <td>1.481013</td>
      <td>2.291139</td>
      <td>3.554430</td>
      <td>5.708861</td>
      <td>10.908861</td>
      <td>10.713924</td>
      <td>10.415190</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.276043</td>
      <td>1.094735</td>
      <td>1.088201</td>
      <td>0.697505</td>
      <td>0.839240</td>
      <td>0.743651</td>
      <td>0.896659</td>
      <td>0.998862</td>
      <td>1.113278</td>
      <td>0.890741</td>
      <td>1.287897</td>
      <td>1.390303</td>
      <td>8.003096</td>
      <td>3.319195</td>
      <td>3.761505</td>
      <td>4.581443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>75.000000</td>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step 15. Summarize all columns of the dataset using the argument include = "all" in the method describe().


```python
student_csv.describe(include='all') # all inclusive NaN
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
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>395</td>
      <td>395</td>
      <td>395.000000</td>
      <td>395</td>
      <td>395</td>
      <td>395</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395</td>
      <td>395</td>
      <td>...</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>GP</td>
      <td>F</td>
      <td>NaN</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>349</td>
      <td>208</td>
      <td>NaN</td>
      <td>307</td>
      <td>281</td>
      <td>354</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>141</td>
      <td>217</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.696203</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.749367</td>
      <td>2.521519</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3.944304</td>
      <td>3.235443</td>
      <td>3.108861</td>
      <td>1.481013</td>
      <td>2.291139</td>
      <td>3.554430</td>
      <td>5.708861</td>
      <td>10.908861</td>
      <td>10.713924</td>
      <td>10.415190</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.276043</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.094735</td>
      <td>1.088201</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.896659</td>
      <td>0.998862</td>
      <td>1.113278</td>
      <td>0.890741</td>
      <td>1.287897</td>
      <td>1.390303</td>
      <td>8.003096</td>
      <td>3.319195</td>
      <td>3.761505</td>
      <td>4.581443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>75.000000</td>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 33 columns</p>
</div>



### Step 16. Summarize only the `Mjob` column


```python
student_csv.Mjob.describe()
```




    count       395
    unique        5
    top       other
    freq        141
    Name: Mjob, dtype: object



### Step 17. What is the mean age of students?


```python
non_round_age = student_csv.age.mean()
round_age = round(non_round_age)
print(round_age)
```

    17.0


### Step 18. What are the three least occurring ages and their frequencies?


```python
student_csv.age.value_counts().tail(3)
```




    20    3
    22    1
    21    1
    Name: age, dtype: int64



### Step 19. Sort the students by `absences` in descending order and show 15 records for the columns 'absences','Fjob','Mjob'


```python
temp1 = student_csv[['absences','Fjob','Mjob']]
temp1
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
      <th>absences</th>
      <th>Fjob</th>
      <th>Mjob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>teacher</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>services</td>
      <td>health</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>teacher</td>
      <td>other</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>health</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>services</td>
      <td>health</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>other</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>other</td>
      <td>health</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>18</th>
      <td>16</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>other</td>
      <td>health</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>other</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>health</td>
      <td>health</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2</td>
      <td>other</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>health</td>
      <td>services</td>
    </tr>
    <tr>
      <th>25</th>
      <td>14</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4</td>
      <td>services</td>
      <td>health</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>29</th>
      <td>16</td>
      <td>teacher</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>365</th>
      <td>4</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>366</th>
      <td>0</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>367</th>
      <td>0</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>368</th>
      <td>0</td>
      <td>services</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>369</th>
      <td>10</td>
      <td>teacher</td>
      <td>other</td>
    </tr>
    <tr>
      <th>370</th>
      <td>4</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>371</th>
      <td>3</td>
      <td>services</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>372</th>
      <td>8</td>
      <td>at_home</td>
      <td>other</td>
    </tr>
    <tr>
      <th>373</th>
      <td>14</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>374</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>375</th>
      <td>2</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>376</th>
      <td>4</td>
      <td>other</td>
      <td>health</td>
    </tr>
    <tr>
      <th>377</th>
      <td>4</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>378</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>379</th>
      <td>17</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>380</th>
      <td>4</td>
      <td>teacher</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>381</th>
      <td>5</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>382</th>
      <td>2</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>383</th>
      <td>0</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>384</th>
      <td>14</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>385</th>
      <td>2</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>386</th>
      <td>7</td>
      <td>at_home</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>387</th>
      <td>0</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>388</th>
      <td>0</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>389</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>390</th>
      <td>11</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>391</th>
      <td>3</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>392</th>
      <td>3</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>393</th>
      <td>0</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>394</th>
      <td>5</td>
      <td>at_home</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
<p>395 rows × 3 columns</p>
</div>




```python
temp2 = temp1.sort_values(['absences'],ascending=False)
temp2
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
      <th>absences</th>
      <th>Fjob</th>
      <th>Mjob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>276</th>
      <td>75</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>183</th>
      <td>56</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>74</th>
      <td>54</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>315</th>
      <td>40</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>307</th>
      <td>38</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>280</th>
      <td>30</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>205</th>
      <td>28</td>
      <td>services</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>103</th>
      <td>26</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>40</th>
      <td>25</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>198</th>
      <td>24</td>
      <td>teacher</td>
      <td>services</td>
    </tr>
    <tr>
      <th>320</th>
      <td>23</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>216</th>
      <td>22</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>277</th>
      <td>22</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>313</th>
      <td>22</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>260</th>
      <td>21</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>311</th>
      <td>20</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>237</th>
      <td>20</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>304</th>
      <td>20</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>118</th>
      <td>20</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>281</th>
      <td>19</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>89</th>
      <td>18</td>
      <td>health</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>234</th>
      <td>18</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>203</th>
      <td>18</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>309</th>
      <td>18</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>123</th>
      <td>18</td>
      <td>other</td>
      <td>health</td>
    </tr>
    <tr>
      <th>379</th>
      <td>17</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>165</th>
      <td>16</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>225</th>
      <td>16</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>29</th>
      <td>16</td>
      <td>teacher</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>133</th>
      <td>16</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>318</th>
      <td>0</td>
      <td>services</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>154</th>
      <td>0</td>
      <td>teacher</td>
      <td>other</td>
    </tr>
    <tr>
      <th>316</th>
      <td>0</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>services</td>
      <td>health</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>at_home</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>310</th>
      <td>0</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>308</th>
      <td>0</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>306</th>
      <td>0</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>303</th>
      <td>0</td>
      <td>health</td>
      <td>health</td>
    </tr>
    <tr>
      <th>302</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>301</th>
      <td>0</td>
      <td>teacher</td>
      <td>other</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>298</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>296</th>
      <td>0</td>
      <td>other</td>
      <td>health</td>
    </tr>
    <tr>
      <th>194</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>291</th>
      <td>0</td>
      <td>services</td>
      <td>health</td>
    </tr>
    <tr>
      <th>160</th>
      <td>0</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>162</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>164</th>
      <td>0</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>167</th>
      <td>0</td>
      <td>services</td>
      <td>health</td>
    </tr>
    <tr>
      <th>168</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>269</th>
      <td>0</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>131</th>
      <td>0</td>
      <td>other</td>
      <td>at_home</td>
    </tr>
  </tbody>
</table>
<p>395 rows × 3 columns</p>
</div>




```python
temp2.head(15)
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
      <th>absences</th>
      <th>Fjob</th>
      <th>Mjob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>276</th>
      <td>75</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>183</th>
      <td>56</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>74</th>
      <td>54</td>
      <td>services</td>
      <td>other</td>
    </tr>
    <tr>
      <th>315</th>
      <td>40</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>307</th>
      <td>38</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>280</th>
      <td>30</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>205</th>
      <td>28</td>
      <td>services</td>
      <td>at_home</td>
    </tr>
    <tr>
      <th>103</th>
      <td>26</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>40</th>
      <td>25</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>198</th>
      <td>24</td>
      <td>teacher</td>
      <td>services</td>
    </tr>
    <tr>
      <th>320</th>
      <td>23</td>
      <td>services</td>
      <td>services</td>
    </tr>
    <tr>
      <th>216</th>
      <td>22</td>
      <td>other</td>
      <td>other</td>
    </tr>
    <tr>
      <th>277</th>
      <td>22</td>
      <td>services</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>313</th>
      <td>22</td>
      <td>other</td>
      <td>services</td>
    </tr>
    <tr>
      <th>260</th>
      <td>21</td>
      <td>other</td>
      <td>services</td>
    </tr>
  </tbody>
</table>
</div>



### Step 20. Select the students with `Fjob` ending at 'h'


```python
t1 = student_csv.Fjob.str.endswith('h')
print(t1)
```

    0      False
    1      False
    2      False
    3      False
    4      False
    5      False
    6      False
    7      False
    8      False
    9      False
    10      True
    11     False
    12     False
    13     False
    14     False
    15     False
    16     False
    17     False
    18     False
    19     False
    20     False
    21      True
    22     False
    23     False
    24      True
    25     False
    26     False
    27     False
    28     False
    29     False
           ...  
    365    False
    366    False
    367    False
    368    False
    369    False
    370    False
    371    False
    372    False
    373    False
    374    False
    375    False
    376    False
    377    False
    378    False
    379    False
    380    False
    381    False
    382    False
    383    False
    384    False
    385    False
    386    False
    387    False
    388    False
    389    False
    390    False
    391    False
    392    False
    393    False
    394    False
    Name: Fjob, Length: 395, dtype: bool



```python
student_csv[t1] # Boolean array indexing 
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
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>teacher</td>
      <td>health</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>12</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>2</td>
      <td>4</td>
      <td>services</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>10</td>
      <td>9</td>
      <td>8</td>
    </tr>
    <tr>
      <th>38</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>R</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>4</td>
      <td>services</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>12</td>
      <td>12</td>
      <td>11</td>
    </tr>
    <tr>
      <th>52</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>4</td>
      <td>2</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>11</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>57</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>teacher</td>
      <td>health</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>14</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>63</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>3</td>
      <td>teacher</td>
      <td>health</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>89</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>teacher</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>18</td>
      <td>8</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>94</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>2</td>
      <td>2</td>
      <td>services</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>11</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>105</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>3</td>
      <td>3</td>
      <td>other</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>109</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>14</td>
      <td>15</td>
      <td>16</td>
    </tr>
    <tr>
      <th>122</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>2</td>
      <td>4</td>
      <td>other</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>169</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>217</th>
      <td>GP</td>
      <td>M</td>
      <td>18</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>3</td>
      <td>3</td>
      <td>services</td>
      <td>health</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>13</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>274</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>2</td>
      <td>4</td>
      <td>at_home</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>278</th>
      <td>GP</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>4</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>15</td>
      <td>9</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>303</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>2</td>
      <td>health</td>
      <td>health</td>
      <td>...</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
    </tr>
    <tr>
      <th>314</th>
      <td>GP</td>
      <td>F</td>
      <td>19</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>health</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>14</td>
      <td>15</td>
      <td>13</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 33 columns</p>
</div>




```python
### Step 21. How many `Mjob` profession names end with 'r'?
```


```python
t1 = student_csv.Mjob.str.endswith('r')

t2 = student_csv[t1]
print(t2)
```

        school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    4       GP   F   16       U     GT3       T     3     3    other     other   
    6       GP   M   16       U     LE3       T     2     2    other     other   
    7       GP   F   17       U     GT3       A     4     4    other   teacher   
    9       GP   M   15       U     GT3       T     3     4    other     other   
    10      GP   F   15       U     GT3       T     4     4  teacher    health   
    13      GP   M   15       U     GT3       T     4     3  teacher     other   
    14      GP   M   15       U     GT3       A     2     2    other     other   
    17      GP   F   16       U     GT3       T     3     3    other     other   
    20      GP   M   15       U     GT3       T     4     3  teacher     other   
    22      GP   M   16       U     LE3       T     4     2  teacher     other   
    23      GP   M   16       U     LE3       T     2     2    other     other   
    26      GP   M   15       U     GT3       T     2     2    other     other   
    29      GP   M   16       U     GT3       T     4     4  teacher   teacher   
    32      GP   M   15       R     GT3       T     4     3  teacher   at_home   
    33      GP   M   15       U     LE3       T     3     3    other     other   
    34      GP   M   16       U     GT3       T     3     2    other     other   
    35      GP   F   15       U     GT3       T     2     3    other     other   
    36      GP   M   15       U     LE3       T     4     3  teacher  services   
    37      GP   M   16       R     GT3       A     4     4    other   teacher   
    40      GP   F   16       U     LE3       T     2     2    other     other   
    41      GP   M   15       U     LE3       T     4     4  teacher     other   
    44      GP   F   16       U     LE3       T     2     2    other   at_home   
    45      GP   F   15       U     LE3       A     4     3    other     other   
    46      GP   F   16       U     LE3       A     3     3    other  services   
    48      GP   M   15       U     GT3       T     4     2  teacher     other   
    54      GP   F   15       U     LE3       A     3     3    other     other   
    55      GP   F   16       U     GT3       A     2     1    other     other   
    57      GP   M   15       U     GT3       T     4     4  teacher    health   
    58      GP   M   15       U     LE3       T     1     2    other   at_home   
    62      GP   F   16       U     LE3       T     1     2    other  services   
    ..     ...  ..  ...     ...     ...     ...   ...   ...      ...       ...   
    339     GP   F   17       R     GT3       A     3     2    other     other   
    341     GP   M   18       U     GT3       T     4     4  teacher  services   
    345     GP   F   18       U     GT3       T     3     2    other  services   
    346     GP   M   18       R     GT3       T     4     3  teacher  services   
    347     GP   M   18       U     GT3       T     4     3  teacher     other   
    349     MS   M   18       R     GT3       T     3     2    other     other   
    350     MS   M   19       R     GT3       T     1     1    other  services   
    353     MS   M   19       R     GT3       T     1     1    other     other   
    356     MS   F   17       R     GT3       T     4     4  teacher  services   
    358     MS   M   18       U     LE3       T     1     1    other  services   
    364     MS   F   17       R     GT3       T     1     2    other  services   
    366     MS   M   18       U     LE3       T     4     4  teacher  services   
    367     MS   F   17       R     GT3       T     1     1    other  services   
    369     MS   F   18       R     GT3       T     4     4    other   teacher   
    372     MS   F   17       U     GT3       T     2     2    other   at_home   
    373     MS   F   17       R     GT3       T     1     2    other     other   
    374     MS   F   18       R     LE3       T     4     4    other     other   
    375     MS   F   18       R     GT3       T     1     1    other     other   
    377     MS   F   18       R     LE3       T     4     4  teacher  services   
    378     MS   F   18       U     GT3       T     3     3    other     other   
    380     MS   M   18       U     GT3       T     4     4  teacher   teacher   
    381     MS   M   18       R     GT3       T     2     1    other     other   
    382     MS   M   17       U     GT3       T     2     3    other  services   
    383     MS   M   19       R     GT3       T     1     1    other  services   
    384     MS   M   18       R     GT3       T     4     2    other     other   
    386     MS   F   18       R     GT3       T     4     4  teacher   at_home   
    388     MS   F   18       U     LE3       T     3     1  teacher  services   
    389     MS   F   18       U     GT3       T     1     1    other     other   
    392     MS   M   21       R     GT3       T     1     1    other     other   
    394     MS   M   19       U     LE3       T     1     1    other   at_home   
    
         ... famrel freetime  goout  Dalc  Walc health absences  G1  G2  G3  
    4    ...      4        3      2     1     2      5        4   6  10  10  
    6    ...      4        4      4     1     1      3        0  12  12  11  
    7    ...      4        1      4     1     1      1        6   6   5   6  
    9    ...      5        5      1     1     1      5        0  14  15  15  
    10   ...      3        3      3     1     2      2        0  10   8   9  
    13   ...      5        4      3     1     2      3        2  10  10  11  
    14   ...      4        5      2     1     1      3        0  14  16  16  
    17   ...      5        3      2     1     1      4        4   8  10  10  
    20   ...      4        4      1     1     1      1        0  13  14  15  
    22   ...      4        5      1     1     3      5        2  15  15  16  
    23   ...      5        4      4     2     4      5        0  13  13  12  
    26   ...      4        2      2     1     2      5        2  12  12  11  
    29   ...      4        4      5     5     5      5       16  10  12  11  
    32   ...      4        5      2     1     1      5        0  17  16  16  
    33   ...      5        3      2     1     1      2        0   8  10  12  
    34   ...      5        4      3     1     1      5        0  12  14  15  
    35   ...      3        5      1     1     1      5        0   8   7   6  
    36   ...      5        4      3     1     1      4        2  15  16  18  
    37   ...      2        4      3     1     1      5        7  15  16  15  
    40   ...      3        3      3     1     2      3       25   7  10  11  
    41   ...      5        4      3     2     4      5        8  12  12  12  
    44   ...      4        3      3     2     2      5       14  10  10   9  
    45   ...      5        2      2     1     1      5        8   8   8   6  
    46   ...      2        3      5     1     4      3       12  11  12  11  
    48   ...      4        3      3     2     2      5        2  15  15  14  
    54   ...      5        3      4     4     4      1        6  10  13  13  
    55   ...      5        3      4     1     1      2        8   8   9  10  
    57   ...      3        2      2     1     1      5        4  14  15  15  
    58   ...      4        3      2     1     1      5        2   9  10   9  
    62   ...      4        4      3     1     1      1        4   8  10   9  
    ..   ...    ...      ...    ...   ...   ...    ...      ...  ..  ..  ..  
    339  ...      4        3      3     2     3      2        4   9  10  10  
    341  ...      4        3      3     2     2      2        0  10  10   0  
    345  ...      5        4      3     2     3      1        7  13  13  14  
    346  ...      5        3      2     1     2      4        9  16  15  16  
    347  ...      5        4      5     2     3      5        0  10  10   9  
    349  ...      2        5      5     5     5      5       10  11  13  13  
    350  ...      5        4      4     3     3      2        8   8   7   8  
    353  ...      4        4      4     3     3      5        4   8   8   8  
    356  ...      4        3      3     1     2      5        4  12  13  13  
    358  ...      3        3      2     1     2      3        4  10  10  10  
    364  ...      3        2      2     1     2      3        0  12  11  12  
    366  ...      4        2      2     2     2      5        0  13  13  13  
    367  ...      5        2      1     1     2      1        0   7   6   0  
    369  ...      3        2      2     4     2      5       10  14  12  11  
    372  ...      3        4      3     1     1      3        8  13  11  11  
    373  ...      3        5      5     1     3      1       14   6   5   5  
    374  ...      5        4      4     1     1      1        0  19  18  19  
    375  ...      4        3      2     1     2      4        2   8   8  10  
    377  ...      5        4      3     3     4      2        4   8   9  10  
    378  ...      4        1      3     1     2      1        0  15  15  15  
    380  ...      3        2      4     1     4      2        4  15  14  14  
    381  ...      4        4      3     1     3      5        5   7   6   7  
    382  ...      4        4      3     1     1      3        2  11  11  10  
    383  ...      4        3      2     1     3      5        0   6   5   0  
    384  ...      5        4      3     4     3      3       14   6   5   5  
    386  ...      4        4      3     2     2      5        7   6   5   6  
    388  ...      4        3      4     1     1      1        0   7   9   8  
    389  ...      1        1      1     1     1      5        0   6   5   0  
    392  ...      5        5      3     3     3      3        3  10   8   7  
    394  ...      3        2      3     3     3      5        5   8   9   9  
    
    [199 rows x 33 columns]



```python
t2.Mjob.value_counts()
```




    other      141
    teacher     58
    Name: Mjob, dtype: int64




```python
t2.Mjob.shape # total
```




    (199,)




```python
t2.Mjob.unique()
```




    array(['other', 'teacher'], dtype=object)




```python
(t2.Mjob.value_counts())
```




    other      141
    teacher     58
    Name: Mjob, dtype: int64




```python
len(t2.Mjob.value_counts())
```




    2




```python
len(t2.Mjob.unique())
```




    2



### Step 22. Who is the guardian for majority of the students? mother, father or other?


```python
t1 = student_csv.guardian.describe()
```


```python
t2 = t1.tail(2)
```


```python
t3 = t2.head(1)
print(t3)
```

    top    mother
    Name: guardian, dtype: object



```python
print(t3[0])
```

    mother



```python
t3[0]
```




    'mother'




```python
t4 = student_csv.guardian.value_counts()
print(t4)
```

    mother    273
    father     90
    other      32
    Name: guardian, dtype: int64



```python
t4.index[0]
```




    'mother'



### Step 23. Find the number of male students


```python
student_csv.sex.describe()
```




    count     395
    unique      2
    top         F
    freq      208
    Name: sex, dtype: object




```python
student_csv.sex.value_counts()
```




    F    208
    M    187
    Name: sex, dtype: int64




```python
t1 = student_csv.sex.value_counts()
```


```python
t1[1]
```




    187


