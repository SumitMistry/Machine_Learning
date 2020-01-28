#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Lab 1 - CKCS 150 Intro to Machine Learning</font>
# ### <font color='red'> Answer the following questions and submit a PDF file on the D2L. </font>
# 
# To complete this lab, you should have the knowledge basics of Python programming:

# ### 1.  Printing elements from a list using index.

# In[31]:


x = [1, "Hello", 3.4]

# In[32]:


print(x)

# # it prints as a LIST : [1, 'Hello', 3.4]
# 

# **Print second element of the list x**

# In[33]:


x[1]

# In[34]:


# nested list
my_list = ["mouse", [8, 4, 6], ['a']]

# In[35]:


secEle = my_list[1]
print(secEle)

# **Print second element of the list my_list using negative index**

# In[36]:


# to print [8, 4, 6] using neg. inde
my_list[-2]

# **Access the element from the list x with index -3**

# In[37]:


# try index -3
# to print 'mouse'
x = my_list[-3]
print(x)

# ### List slices
# The slice operator also works on lists:

# #### 2.  Use slice operator to print list contents.
# **Start, end.** 
# In slicing, <br> if you omit the first index, it means "start" of the list<br>
# if you omit the second index, it means "end" of the list

# In[38]:


my_list = ['p', 'r', 'o', 'g', 'r', 'a', 'm', 'i', 'z']

# **Read elements 3rd to 5th from my_list**

# In[39]:


# try [2:5]
my_list[2:5]

# #### Skip Step.
# There is a third, optional index used in slicing: the step. If the step is two, we advance two places after an element is copied. So we can skip over elements.
# 
# (example.start, example.stop, example.step)

# **Use [::2] to print elements from my_list, start through end, skipping ahead 2 places each time.**

# In[40]:


# try ::2   #Start through end, skipping ahead 2 places each time.
# my_list = ['p','r','o','g','r','a','m','i','z']
# get output of "['p', 'o', 'r', 'm', 'z']"


my_list[::1]

# In[41]:


my_list[::2]

# In[42]:


my_list[::3]

# In[43]:


my_list[::4]

# **Use [::-1]  to print all elements from my_list in reverse order.**
# <br>**Negative values** also work to make a copy of the same list in reverse order:

# In[44]:


# try ::-1  # this will print all elements in reverse order
# get this o/p ['z', 'i', 'm', 'a', 'r', 'g', 'o', 'r', 'p']
# my_list = ['p','r','o','g','r','a','m','i','z']

my_list[:]  # same as given
my_list[::]  # same as given

my_list[:-1]  # cuts the last char == ['p', 'r', 'o', 'g', 'r', 'a', 'm', 'i']
my_list[::-1]  # reverse of the given str

# ### Strings
# A string is a sequence of characters. You can access the characters one at a time with the bracket operator:
# https://docs.python.org/2/library/string.html

# In[45]:


astring = "Hello world!"

# #### 3. Functions for manipulating the strings

# **Write Python script to print length of the string: astring**

# In[46]:


# O/P should be 12

print(len(astring))

# **Write Python script to print index of o in the string: astring**

# In[47]:


# index function
# O/P ==4
# astring = "Hello world!"
# nice article https://realpython.com/python-strings/
# print(index(astring))

astring.find('o')

# **Using the count() function to print the occurences of character l in the string astring**

# In[48]:


# count function
astring.count('l')

# **Write Python script to print astring in upper case**

# In[49]:


# upper
# astring.swapcase()
astring.upper()

# **Write Python script to print astring in lower case**

# In[50]:


# lower
astring.lower()

# **Write Python script to test if astring starts with "Hello" (startswith)**

# In[51]:


# startswith
astring.startswith('Hello')  # or astring.startswith("Hello") both fine

# **Write Python script to test if astring ends with "asdf" (endswith)**

# In[52]:


# endswith
astring.startswith('asdf')

# **Write Python script to split the string astring as ['Hello', 'world!']**

# In[53]:


# split
# O/P must be ['Hello', 'world!']
astring.split(' ')  # I am splitting with ' ' whitespace char

# **Write Python script to split the string astring as ['Hell', ' w', 'rld!']**

# In[54]:


# split("o")
# O/P ['Hell', ' w', 'rld!']

astring.split('o')

# ### Dictionary
# Dictionaries are sometimes found in other languages as “associative memories” or “associative arrays”. Unlike sequences, which are indexed by a range of numbers, dictionaries are indexed by keys, which can be any immutable type; strings and numbers can always be keys.
# https://docs.python.org/2/tutorial/datastructures.html#dictionaries

# In[55]:


myDict = {"university": "Ryerson", "course": "Python", "year": 2020}

# **Read the value of the "university" key: from myDict using: myDict["university"]**

# In[56]:


myDict['university']  ##key[]

# **Read the value of the "university" key using get function: myDict.get("university")**

# In[57]:


# O/P = 'Ryerson'
myDict.get("university")  # get() function ()

# **Add an item to 'myDict' with key as 'class' and value as 'Friday' and print contents of myDict.**

# In[58]:


# O/P {'university': 'Ryerson', 'course': 'Python', 'year': 2020, 'class': 'Friday'}
# myDict == {'university': 'Ryerson', 'course': 'Python', 'year': 2020, 'class': 'Friday'}
myDict['class'] = 'Friday'

# In[59]:


print(myDict)

# **Print all keys and values in the dictionary, one by one using a for loop**<br>
# **Possible solutions**<br>
# for x in myDict:<br>
#     print('Key is :', x, ' and Vlaue is : ',myDict[x])<br>
#     <br>
# OR<br>
# for x, y in myDict.items():<br>
#     print('Key is :', x, ' and Vlaue is : ', y)<br>

# In[61]:


myDict = {'university': 'Ryerson', 'course': 'Python', 'year': 2020, 'class': 'Friday'}

# In[62]:


for x in myDict:
    print('#Key:', x, ' Value:', myDict[x])

# In[63]:


myDict

# In[64]:


# OP ==
# Key is : university  and Vlaue is :  Ryerson
# Key is : course  and Vlaue is :  Python
# Key is : year  and Vlaue is :  2020
# Key is : class  and Vlaue is :  Friday


# **Delete the item 'class': 'Friday' from myDict using the operator del**

# In[65]:


del myDict['class']

# **Delete the item 'year': 2020 from myDict using the function pop**

# In[66]:


myDict.pop('year')


# **Write a function to check whether a number is even or odd**

# In[67]:


def evenOdd(x):
    if x % 2 == 0:
        print("num is even")
    else:
        print("num is odd")


y = int(input("Enter # :"))
evenOdd(y)

# In[68]:


evenOdd(9)
evenOdd(2)

# **Suppose we have profits made by an organisation as follows:**

# In[83]:


months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
sales = [66, 90, 68, 59, 76, 60, 88, 71, 81, 65, 94, 74]
data = list(map(lambda x, y: (x, y), months, sales))
print(data)

# In[84]:


data2 = list(map(lambda x, y: {x, y}, months, sales))  # making myDict K:V dictionary
print(data2)

# In[85]:


data3 = list(map(lambda x, y: [x, y], months, sales))  # making List
print(data3)

# In[86]:


import numpy

arr1 = numpy.array(data)
arr1.T[1]

# **Find the names of months from data where profits are greater than 72**

# In[87]:


data1 = list(filter(lambda y: (y[1] > 72), data))
print(data1)

# In[95]:


arr1 = numpy.array(data1)
print(arr1.T[0])


