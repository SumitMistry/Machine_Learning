# <font color='blue'>Week 1 - Machine Learning Intro</font>
Used:
Python and its libraries.
Jupyter Notebook.

### 1.  Printing elements from a list using index.


```python
x = [1, "Hello", 3.4]
```


```python
print(x)
```

    [1, 'Hello', 3.4]


# it prints as a LIST : [1, 'Hello', 3.4]


**Print second element of the list x**


```python
x[1]
```




    'Hello'




```python
# nested list
my_list = ["mouse", [8, 4, 6], ['a']]
```


```python
secEle = my_list[1]
print(secEle)
```

    [8, 4, 6]


**Print second element of the list my_list using negative index**


```python
# to print [8, 4, 6] using neg. inde
my_list[-2]
```




    [8, 4, 6]



**Access the element from the list x with index -3**


```python
#try index -3
# to print 'mouse'
x = my_list[-3]
print(x)


```

    mouse


### List slices
The slice operator also works on lists:

#### 2.  Use slice operator to print list contents.
**Start, end.** 
In slicing, <br> if you omit the first index, it means "start" of the list<br>
if you omit the second index, it means "end" of the list


```python
my_list = ['p','r','o','g','r','a','m','i','z']
```

**Read elements 3rd to 5th from my_list**


```python
#try [2:5]
my_list[2:5]

```




    ['o', 'g', 'r']



#### Skip Step. 
There is a third, optional index used in slicing: the step. If the step is two, we advance two places after an element is copied. So we can skip over elements.

(example.start, example.stop, example.step)

**Use [::2] to print elements from my_list, start through end, skipping ahead 2 places each time.**


```python
#try ::2   #Start through end, skipping ahead 2 places each time.
#my_list = ['p','r','o','g','r','a','m','i','z']
# get output of "['p', 'o', 'r', 'm', 'z']"


my_list[::1]

```




    ['p', 'r', 'o', 'g', 'r', 'a', 'm', 'i', 'z']




```python
my_list[::2]
```




    ['p', 'o', 'r', 'm', 'z']




```python

my_list[::3]
```




    ['p', 'g', 'm']




```python

my_list[::4]
```




    ['p', 'r', 'z']



**Use [::-1]  to print all elements from my_list in reverse order.**
<br>**Negative values** also work to make a copy of the same list in reverse order:


```python
#try ::-1  # this will print all elements in reverse order
# get this o/p ['z', 'i', 'm', 'a', 'r', 'g', 'o', 'r', 'p']
#my_list = ['p','r','o','g','r','a','m','i','z']

my_list[:] # same as given
my_list[::] # same as given

my_list[:-1] # cuts the last char == ['p', 'r', 'o', 'g', 'r', 'a', 'm', 'i']
my_list[::-1] # reverse of the given str
```




    ['z', 'i', 'm', 'a', 'r', 'g', 'o', 'r', 'p']



### Strings
A string is a sequence of characters. You can access the characters one at a time with the bracket operator:
https://docs.python.org/2/library/string.html


```python
astring = "Hello world!"
```

#### 3. Functions for manipulating the strings

**Write Python script to print length of the string: astring**


```python
# O/P should be 12

print(len(astring))
```

    12


**Write Python script to print index of o in the string: astring**


```python
#index function
# O/P ==4
# astring = "Hello world!"
# nice article https://realpython.com/python-strings/
#print(index(astring))

astring.find('o')

```




    4



**Using the count() function to print the occurences of character l in the string astring**


```python
#count function
astring.count('l')
```




    3



**Write Python script to print astring in upper case**


```python
#upper
#astring.swapcase()
astring.upper()

```




    'HELLO WORLD!'



**Write Python script to print astring in lower case**


```python
#lower
astring.lower()

```




    'hello world!'



**Write Python script to test if astring starts with "Hello" (startswith)**


```python
#startswith
astring.startswith('Hello') # or astring.startswith("Hello") both fine
```




    True



**Write Python script to test if astring ends with "asdf" (endswith)**


```python
#endswith
astring.startswith('asdf') 
```




    False



**Write Python script to split the string astring as ['Hello', 'world!']**


```python
#split
# O/P must be ['Hello', 'world!']
astring.split(' ') # I am splitting with ' ' whitespace char
```




    ['Hello', 'world!']



**Write Python script to split the string astring as ['Hell', ' w', 'rld!']**


```python
#split("o")
# O/P ['Hell', ' w', 'rld!']

astring.split('o')

```




    ['Hell', ' w', 'rld!']



### Dictionary
Dictionaries are sometimes found in other languages as “associative memories” or “associative arrays”. Unlike sequences, which are indexed by a range of numbers, dictionaries are indexed by keys, which can be any immutable type; strings and numbers can always be keys.
https://docs.python.org/2/tutorial/datastructures.html#dictionaries


```python
myDict = {"university": "Ryerson",   "course": "Python",  "year": 2020}
```

**Read the value of the "university" key: from myDict using: myDict["university"]**


```python
myDict['university'] ##key[]
```




    'Ryerson'



**Read the value of the "university" key using get function: myDict.get("university")**


```python
# O/P = 'Ryerson'
myDict.get("university") # get() function ()
```




    'Ryerson'



**Add an item to 'myDict' with key as 'class' and value as 'Friday' and print contents of myDict.**


```python
#O/P {'university': 'Ryerson', 'course': 'Python', 'year': 2020, 'class': 'Friday'}
# myDict == {'university': 'Ryerson', 'course': 'Python', 'year': 2020, 'class': 'Friday'}
myDict['class'] = 'Friday'

```


```python
print(myDict)
```

    {'course': 'Python', 'university': 'Ryerson', 'class': 'Friday', 'year': 2020}


**Print all keys and values in the dictionary, one by one using a for loop**<br>
**Possible solutions**<br>
for x in myDict:<br>
    print('Key is :', x, ' and Vlaue is : ',myDict[x])<br>
    <br>
OR<br>
for x, y in myDict.items():<br>
    print('Key is :', x, ' and Vlaue is : ', y)<br>


```python
myDict = {'university': 'Ryerson', 'course': 'Python', 'year': 2020, 'class': 'Friday'}
```


```python
for x in myDict:
    print('#Key:', x , ' Value:',myDict[x])
```

    ('#Key:', 'course', ' Value:', 'Python')
    ('#Key:', 'university', ' Value:', 'Ryerson')
    ('#Key:', 'class', ' Value:', 'Friday')
    ('#Key:', 'year', ' Value:', 2020)



```python
myDict
```




    {'class': 'Friday', 'course': 'Python', 'university': 'Ryerson', 'year': 2020}




```python
# OP ==
#Key is : university  and Vlaue is :  Ryerson
#Key is : course  and Vlaue is :  Python
#Key is : year  and Vlaue is :  2020
#Key is : class  and Vlaue is :  Friday

```

**Delete the item 'class': 'Friday' from myDict using the operator del**


```python
del myDict['class']
```

**Delete the item 'year': 2020 from myDict using the function pop**


```python
myDict.pop('year')
```




    2020



**Write a function to check whether a number is even or odd**


```python
def evenOdd(x):
    if x%2==0:
        print("num is even")
    else:
        print("num is odd")

y = int(input("Enter # :"))
evenOdd(y)
```

    num is odd



```python
evenOdd(9)
evenOdd(2)
```

    num is odd
    num is even


**Suppose we have profits made by an organisation as follows:**


```python
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
sales = [66, 90, 68, 59, 76, 60, 88, 71, 81, 65, 94, 74]
data = list(map(lambda x, y: (x, y), months, sales))
print(data)
```

    [('January', 66), ('February', 90), ('March', 68), ('April', 59), ('May', 76), ('June', 60), ('July', 88), ('August', 71), ('September', 81), ('October', 65), ('November', 94), ('December', 74)]



```python
data2 = list(map(lambda x,y : {x,y}, months, sales)) #making myDict K:V dictionary
print(data2)
```

    [set(['January', 66]), set(['February', 90]), set([68, 'March']), set(['April', 59]), set(['May', 76]), set([60, 'June']), set([88, 'July']), set(['August', 71]), set(['September', 81]), set([65, 'October']), set(['November', 94]), set(['December', 74])]



```python
data3 = list(map(lambda x,y : [x,y], months, sales)) #making List
print(data3)
```

    [['January', 66], ['February', 90], ['March', 68], ['April', 59], ['May', 76], ['June', 60], ['July', 88], ['August', 71], ['September', 81], ['October', 65], ['November', 94], ['December', 74]]



```python
import numpy
arr1 = numpy.array(data)
arr1.T[1]
```




    array(['66', '90', '68', '59', '76', '60', '88', '71', '81', '65', '94',
           '74'], dtype='|S9')



**Find the names of months from data where profits are greater than 72**


```python
data1 = list(filter(lambda y:(y[1]>72), data))
print(data1)
```

    [('February', 90), ('May', 76), ('July', 88), ('September', 81), ('November', 94), ('December', 74)]



```python
arr1 = numpy.array(data1)
print(arr1.T[0])
```

    ['February' 'May' 'July' 'September' 'November' 'December']



