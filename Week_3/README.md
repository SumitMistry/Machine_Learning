# <font color='blue'>Week 3 - Machine Learning Intro</font>
Used:
Python and its libraries; NumPy and Pandas library.
Jupyter Notebook.


Q 1. Modify the class Student by adding:

- Two more attributes 
- Two more methods
- Instantiate an object of the class
- To test the class, call methods with the object


```python
class Student:
    """A class representing a student."""
    def __init__(self,n,a):
        self.full_name = n
        self.age = a
    def get_age(self):
        return self.age
```

### Modified class


```python
class Student:
    
   #initialize
    def __init__(self, nm, ag, x, ccps, bcty):
        self.full_name =nm
        self.age =ag
        self.sex =x
        self.course =ccps
        self.birth_city = bcty
        
   #Setter functions examples:     
    def setAge(self, ag):
        self.age = ag # (or) return self.age (if) def set_age(self):
    def setSex(self, x):
        self.sex = x
    def set_birth_city(self, bcty):
        self.birth_city = bcty
    
   #Getter functions examples:
    def getInfo(self):
        print("Student Name:", self.full_name, 
              "Age:", self.age,
              "Gender:", self.sex,
              "is taking the Course:", self.course,
              "BirthCity:", self.birth_city)
    def get_age(self):
        return ("My age is:", self.age)
```


```python
try1 = Student('Nick',"555","M","Intro to Python","HK")

#I wanted to change values...#yes, you can overwrite!
try1.setAge(25) 
try1.setSex('Male')

try1.getInfo()

```

    Student Name: Nick Age: 25 Gender: Male is taking the Course: Intro to Python BirthCity: HK
    


```python
try1 = Student('Nick', 555, 'F', 'Intro to Python', 'JP')

#I wanted to change values...#yes, you can overwrite!
try1.setAge(25) 
try1.setSex('Male')
try1.set_birth_city('MIAMI')

try1.getInfo()
#additonal getter_method call
try1.get_age()
```
    Student Name: Nick Age: 25 Gender: Male is taking the Course: Intro to Python BirthCity: MIAMI
    
    ('My age is:', 25)

```python
#test the class
obj = Student("Nick",25,"Male","Intro to Python", "NY")
obj.getInfo()
```

    Student Name: Nick Age: 25 Gender: Male is taking the Course: Intro to Python BirthCity: NY 

Q 2. Create a class of your choice having at least three attributes and two methods. Test the class by instantiating the objects and calling the methods.

```python
#new class

class Account:
    
    #initialize
    def __init__(self, nam, acNum, sn, opBal ):
        self.name = nam
        self.setAcNum(acNum)
        self.sin= sn
        self.setOpBal(opBal)
    
    #setter functions
    def setAcNum(self, acn):
         self.account_Number = '2020#1234:' + str(acn) #defauult format
    
    def setOpBal(self, op):
        self.opening_balance = 'USD: ' + str (op + 1000) + '$' # bank courtesy new customer
    
    #getter fucntions
    def getAccountInfo(self):
        print ("Ac Holder Name :", self.name,
             "\nAc number      :", self.account_Number,
             "\nSin Num        :", self.sin,
             "\nOpening Balance:", self.opening_balance  )
```


```python
accObj1 = Account("Sumit", 5252, 587587587,40)
accObj1.getAccountInfo()
```

    Ac Holder Name : Sumit 
    Ac number      : 2020#1234:5252 
    Sin Num        : 587587587 
    Opening Balance: USD: 1040$

```python
#test the class

```
Q 3. Create an Employee class. The Employee class has four instance variables and methods
- name, age, designation and salary. 
- methods (you can select methods of your choice)
- Instantiate objects and call methods to test the class

```python
#new class

class Employee:
    
    def __init__(self, nam, age, title, compensa):
        self.name = nam
        self.age = age
        self.designation =title
        self.salary = compensa
        
    def setName(self, nn):
        self.name = str(nn) + ' ' + 'Patel'
    
    def setAge(self,yy):
        self.age =yy
    
    def getEmpInfo(self):
        print("Employee name: ", self.name ,  
             "\nAge: ", self.age ,
              "\nTitle: ", self.designation ,
             "\nSalary: ", self.salary     )
        
```


```python
empObj1 = Employee("Sumit", 9, "Dev", "10k")
empObj1.getEmpInfo()
```
    Employee name:  Sumit 
    Age:  9 
    Title:  Dev 
    Salary:  10k
    
```python
empObj1.setName("Mit")
empObj1.setAge(str(25) + ' yrs')

empObj1.getEmpInfo()
```

    Employee name:  Mit Patel 
    Age:  25 yrs 
    Title:  Dev 
    Salary:  10k
    
```python
#test the class
%autosave 60
```

    Autosaving every 60 seconds
    
**Import necessary libraries**


```python
import pandas 
import numpy
import matplotlib.pyplot as matPltLibPyPlt
import seaborn as sns
%matplotlib inline
```

**Read the 'house-prices.csv' file**

It's not here:
https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/datasets/data/boston_house_prices.csv

https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/datasets/data/

It's in local folder of week-3


```python
from sklearn.datasets import load_boston
myBoston = load_boston()
```


```python
# housingCSV = pandas.read_csv('house-prices.csv')
# housingCSV
```


```python
myBoston
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
     'DESCR': "Boston House Prices dataset\n===========================\n\nNotes\n------\nData Set Characteristics:  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive\n    \n    :Median Value (attribute 14) is usually the target\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttp://archive.ics.uci.edu/ml/datasets/Housing\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n**References**\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)\n"}


```python
myBoston.keys()
```
    dict_keys(['data', 'target', 'feature_names', 'DESCR'])

```python
myBoston.data.shape
```
    (506, 13)

```python
# the target data usually known as Y
```


```python
myBoston.target.shape
```

    (506,)

```python
myBoston.data
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
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(myBoston.data, myBoston.target, test_size = 0.3, random_state=42)
```


```python
x_train[0]
```
    array([2.9850e-02, 0.0000e+00, 2.1800e+00, 0.0000e+00, 4.5800e-01,
           6.4300e+00, 5.8700e+01, 6.0622e+00, 3.0000e+00, 2.2200e+02,
           1.8700e+01, 3.9412e+02, 5.2100e+00])

```python
x_test[0]
```
    array([9.1780e-02, 0.0000e+00, 4.0500e+00, 0.0000e+00, 5.1000e-01,
           6.4160e+00, 8.4100e+01, 2.6463e+00, 5.0000e+00, 2.9600e+02,
           1.6600e+01, 3.9550e+02, 9.0400e+00])

**Print few records from the dataFrame**

```python
myHouseData = pandas.read_csv("house-prices.csv")
myHouseData.head(5)
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
      <th>Home</th>
      <th>Price</th>
      <th>SqFt</th>
      <th>Bedrooms</th>
      <th>Bathrooms</th>
      <th>Offers</th>
      <th>Brick</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>114300</td>
      <td>1790</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>No</td>
      <td>East</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>114200</td>
      <td>2030</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>No</td>
      <td>East</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>114800</td>
      <td>1740</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>No</td>
      <td>East</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>94700</td>
      <td>1980</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>No</td>
      <td>East</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>119800</td>
      <td>2130</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>No</td>
      <td>East</td>
    </tr>
  </tbody>
</table>
</div>

```python
myHouseData.Neighborhood.shape
```
    (128,)

**Use area in square feet - SqFt (name this feature as X) to predict the house Price (name this feature as y).**


```python
myHouseData.keys()
```
    Index(['Home', 'Price', 'SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick',
           'Neighborhood'],
          dtype='object')

```python
x = myHouseData.SqFt.values

```


```python
# x = x[ numpy.newaxis]
# print(x)
# x
```


```python
x = x[ :, numpy.newaxis]
x
# x 
```
    array([[1790],
           [2030],
           [1740],
           [1980],
           [2130],
           [1780],
           [1830],
           [2160],
           [2110],
           [1730],
           [2030],
           [1870],
           [1910],
           [2150],
           [2590],
           [1780],
           [2190],
           [1990],
           [1700],
           [1920],
           [1790],
           [2000],
           [1690],
           [1820],
           [2210],
           [2290],
           [2000],
           [1700],
           [1600],
           [2040],
           [2250],
           [1930],
           [2250],
           [2280],
           [2000],
           [2080],
           [1880],
           [2420],
           [1720],
           [1740],
           [1560],
           [1840],
           [1990],
           [1920],
           [1940],
           [1810],
           [1990],
           [2050],
           [1980],
           [1700],
           [2100],
           [1860],
           [2150],
           [2100],
           [1650],
           [1720],
           [2190],
           [2240],
           [1840],
           [2090],
           [2200],
           [1610],
           [2220],
           [1910],
           [1860],
           [1450],
           [2210],
           [2040],
           [2140],
           [2080],
           [1950],
           [2160],
           [1650],
           [2040],
           [2140],
           [1900],
           [1930],
           [2280],
           [2130],
           [1780],
           [2190],
           [2140],
           [2050],
           [2410],
           [1520],
           [2250],
           [1900],
           [1880],
           [1930],
           [2010],
           [1920],
           [2150],
           [2110],
           [2080],
           [2150],
           [1970],
           [2440],
           [2000],
           [2060],
           [2080],
           [2010],
           [2260],
           [2410],
           [2440],
           [1910],
           [2530],
           [2130],
           [1890],
           [1990],
           [2110],
           [1710],
           [1740],
           [1940],
           [2000],
           [2010],
           [1900],
           [2290],
           [1920],
           [1950],
           [1920],
           [1930],
           [1930],
           [2060],
           [1900],
           [2160],
           [2070],
           [2020],
           [2250]], dtype=int64)
```python
y = myHouseData.Price.values
y
```

    array([114300, 114200, 114800,  94700, 119800, 114600, 151600, 150700,
           119200, 104000, 132500, 123000, 102600, 126300, 176800, 145800,
           147100,  83600, 111400, 167200, 116200, 113800,  91700, 106100,
           156400, 149300, 137000,  99300,  69100, 188000, 182000, 112300,
           135000, 139600, 117800, 117100, 117500, 147000, 131300, 108200,
           106600, 133600, 105600, 154000, 166500, 103200, 129800,  90300,
           115900, 107500, 151100,  91100, 117400, 130800,  81300, 125700,
           140900, 152300, 138100, 155400, 180900, 100900, 161300, 120500,
           130300, 111100, 126200, 151900,  93600, 165600, 166700, 157600,
           107300, 125700, 144200, 106900, 129800, 176500, 121300, 143600,
           143400, 184300, 164800, 147700,  90500, 188300, 102700, 172500,
           127700,  97800, 143100, 116500, 142600, 157100, 160600, 152500,
           133300, 126800, 145500, 171000, 103200, 123100, 136800, 211200,
            82300, 146900, 108500, 134000, 117000, 108700, 111600, 114900,
           123600, 115700, 124500, 102500, 199500, 117800, 150200, 109700,
           110400, 105600, 144800, 119700, 147900, 113500, 149900, 124600],
          dtype=int64)


**Import Linear Regression library**


```python
from sklearn.linear_model import LinearRegression
```

## Creating and Training the Model

**Instantiate an object of the LinearRegression class**


```python
myLinearModel = LinearRegression()
```

**Fit the model to X and y**


```python
myLinearModel.fit(x,y)
```

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

**Print the coefficient**


```python
myLinearModel.coef_
```
    array([70.22631824])

**Print the intercept**


```python
myLinearModel.intercept_
```
    -10091.12990912312

```python
myLinearModel
```

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

**Predict the price for a house with area 1795 Square feet**


```python
ar1 = myLinearModel.predict([[1795]])
ar1
```
    array([115965.11133686])

**Write down the linear regression equation for the built model**


```python
# y = mx + c
# y = (-10091.129909123149) + 70.22631824(x)
```

**Predict the price for a house with area 1795 Square feet using the linear regression equation**


```python
y = -10091.129909123149 + (70.22631824*1795)
y
```
    115965.11133167685

```python

```