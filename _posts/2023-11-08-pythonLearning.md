---
title: 'Pyhon Learning'
date: 2015-08-14
permalink: /posts/2023/11/pyhonLearning/
tags:
  - Pyhon
---

---
title: python小知识点
author: haomingx
avatar: /images/favicon.png
authorDesc: 不断折腾
comments: true
date: 2023-01-04 11:26:21
authorLink:
authorAbout:
categories:
series:
tags: python小知识
keywords:
description:
photos:

---

###python小知识点

#### 可变类型拷贝

```python
import copy
with open("raw data.txt","r")as f:
	raw data f.readlines()
raw_data [[int(i)for i in data.strip().split("")for data in raw data]
#[[1,2,3,4,5],[6,7,8,9,0],[2,2,2,2,2],[4,4,4,4,4],[6,7,1,9,8]]

raw_data_copy = rawdata.copy() # 一维数据copy样拷贝还会改变原数据
raw_data_copy = copy.deepcopy(raw_data)  #多维数据拷贝
fraw_data_copy[0][0]99999
print(raw_data)
#[1,2,3,4,5],[6,7,8,9,0],[2,2,2,2,2],[4,4,
4,4,4],6,7,1,9,0]
```

函数默认参数为可变类型

```python
def add_fruit(fruit,fruit list=[]):
	fruit_list.append(fruit)
	print(fruit_list)
fruits=['banana','apple']
add_fruit('watermelon',fruits)
['banana','apple','watermelon']
```

```python
def add_fruit(fruit,fruit list=[]):
	fruit_list.append(fruit)
	print(fruit_list)

add fruit('watermelon') #['watermelon']
add fruit('banana')  #['watermelon', 'banana']
```

由于可变，所以fruit_list会在前面的基础上调用

```python
def add_fruit(fruit,fruit list=None):
    if fruit_list is None:
        fruit_list = []
	fruit_list.append(fruit)
	print(fruit_list)

print(add_fruit.__defaults__ ) # (None, )
add_fruit('watermelon') # ['watermelon']
print(add_fruit.__defaults__ ) # (None, )
add_ fruit( 'banana' ) #['banana']
print(add_fruit.__defaults__ ) # (None, )
```

所以尽可能避免将可变类型作为默认参数，而是在内部判断



#### 函数默认值属性

**在定义时就被确定了**

```python
import time
from datetime import datetime
def display_time(data=datetime.now( )) :
	print(data.strftime('%B %d, %Y %H:%M:%S'))
print (display_time.__defaults__ ) # (datetime.datetime (2022,11,26,17, 4, 53, 360783), )
display_time() # November 26, 2022 17 :04:53
time.sleep(2)
display_time() # November 26, 2022 17 :04:53 
time.sleep(2)
display_time() # November 26, 2022 17 :04:53

```

修改

```python
def display_time(data=None):
    if data is None:
        data = datatime.Now()
	print(data.strftime('%B %d, %Y %H:%M:%S'))
```

#### 下划线的含义

· 单引号下划线： `_var`

单下划线是一种Python命名约定，表示某个名称是供内部使用的，只是对程序员的提示，不会有多余的效果

· 单尾划线： `var_`

一个变量最合适的名字已经被一个关键字代替了等情况，打破命名冲突

· 双领先下划线： `__var`

双下划线前缀导致Python解释器重写属性名，以避免子类中的命名冲突

```python
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 23
        self.__baz = 23
        
>>> t = Test()
>>> dir(t)
['_Test__baz', '__class__', '__delattr__', '__dict__', '__dir__',
 '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
 '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__',
 '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
 '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
 '__weakref__', '_bar', 'foo']
>>> t.__dict__
{'foo': 11, '_bar': 23, '_Test__baz': 23}
```

补充`__dict__`的博客：http://c.biancheng.net/view/2374.html

可以看到`foo`和`_bar`的变量属性均未被修改，但是`__baz`被修改为`_Test__baz`,为了保护变量不被子类覆盖

```python
class ManglingTest:
    def __init__(self):
        self.__mangled = 'hello'

    def get_mangled(self):
        return self.__mangled

>>> ManglingTest().get_mangled()
'hello'
>>> ManglingTest().__mangled
AttributeError: "'ManglingTest' object has no attribute '__mangled'"
```

· 领先和落后双下划线： `__var__`

Python中存在一些特殊的方法，有些方法以双下划线`__`开头和结尾，它们是Python的魔法函数，比如`__init__()`和`__str__`等等。**不用要这种方式命名自己的变量或者函数**

---

魔法函数是指类内部以双下划线开头，并且以双下划线结尾的函数，在特定时刻，Python会自动调用这些函数

---

https://www.cnblogs.com/chenhuabin/p/13752770.html#_label0

· 单下划线： `_`

一个单独的下划线有时用作一个名称，表示一个变量是临时的或是不重要的

#### filter、map、reduce、apply

- filter(function，sequence)

**过滤掉序列中不符合函数条件的元素**,返回迭代器,需要list转列表等

```python
x = [1,2,3,4,5]
print(list(filter(lambda x:x%2==0,x))) # 找出偶数。python3.*之后filter函数返回的不再是列表而是迭代器，所以需要用list转换。
# 输出：
[2, 4]
```

- map(function,iterable1,iterable2)

**求一个序列或者多个序列进行函数映射之后的值**，就该想到map这个函数,返回迭代器

```python
x = [1,2,3,4,5]
y = [2,3,4,5,6]
print(list(map(lambda x,y:(x*y)+2,x,y)))
# 输出：
[4, 8, 14, 22, 32]
```

- reduce（function，iterable）

**对一个序列进行压缩运算，得到一个值**

```python
from functools import reduce
arr = [2,3,4,5,6]
reduce(lambda x,y: x + y,arr) # 直接返回一个值
# 20
```

其计算原理：
先计算头两个元素：f(2, 3)，结果为5；
再把结果和第3个元素计算：f(5, 4)，结果为9；
再把结果和第4个元素计算：f(9, 5)，结果为14；
再把结果和第5个元素计算：f(14, 6)，结果为20；
由于没有更多的元素了，计算结束，返回结果20。

- apply(function,axis)

pandas中的函数，eg:   `data.apply(lambda x:x*10)`

---

filter和map都是python内置的函数，可以直接调用，reduce在functools模块，apply在pandas模块

---

#### yield

带有 yield 的函数在 Python 中被称之为 generator（生成器）

博客：https://blog.csdn.net/mieleizhi0522/article/details/82142856

#### python数值

- **整型(int)** - 通常被称为是整型或整数，是正或负整数，不带小数点。Python3 整型是没有限制大小的，可以当作 Long 类型使用，所以 Python3 没有 Python2 的 Long 类型。布尔(bool)是整型的子类型。
- **浮点型(float)** - 浮点型由整数部分与小数部分组成，浮点型也可以使用科学计数法表示（2.5e2 = 2.5 x 102 = 250）
- **复数( (complex))** - 复数由实数部分和虚数部分构成，可以用a + bj,或者complex(a,b)表示， 复数的实部a和虚部b都是浮点型。



#### list.append()无返回值

```python
>>> case1 = [1, 2, 3]

>>> print(case1.append(1))

None

# list.append没有返回值

# 还有clear, insert, sort, reverse, remove, extend
```

所以`case = case.append(1)`这样的操作不可行



#### 列表是可变的

```python
case = [9, 8, 8, 3, 3, 1]

for i in case:
   if i % 2 == 0:
     case.remove(i)
     
print(case)

>> [9, 8, 3, 3, 1]
```

因为列表是可变对象，当第一个8被删除后，第二个8补上了前面的位置，自然而然就被跳过了

修改

```python
case = [9, 8, 8, 3, 3, 1]
case1 = [x for x in case if x%2 != 0]
```



#### 字符串常量用空格连接

表示字符串合并

```python
case = 'a' 'b'
print(case)

# ab
```



#### tuple只有一个元素要在末尾加逗号

```python
>>> isinstance(('bilibili'), tuple)
>>> isinstance(('bilibili',), tuple)

False
True

a = ('bilibili')
for i in a:
    print(i)
b
i
l
i
b
i
l
i

b = ('bilibli',)
for i in b:
    print(i)
bilibli
```



#### if-else表达式优先级高于逗号

```python
>>> x, y = (10, 10) if True else None, None
>>> x
(10, 10)
>>> y
None
```



#### 使用enumerate遍历：

```python
data = [1, 4, 5, 7, 9]
for idx, num in enumerate(data):
    if num % 2:
        data[idx] = 0
print(data)

>>> [0, 4, 0, 0, 0]
```



#### 字典的get方法

在工程文件中经常会注意到使用get方法，避免因键不存在而引起的程序崩溃，若索引不到，将返回在第二个位置定义的参数

```python
data = {"name" : "sds", "age" : "18"}
uid = data.get("uid", "6688")
print(uid)

6688
```



#### f-string新格式化方法

Python3.6开始支持的新的格式化操作，相比以前更简洁方便。

```python
i = 9
data = f"{i} * {i} = {i * i}"
print(data)

>>> 9 * 9 = 81
```

我们也常常这样

```python
name = 'xhm'
print('hello,'+name+'!')
```

改为

```python
print(f'hello,{name}!')
# hello,xhm!
```



#### 合并两个字典

```python
data_1 = {"name" : "sds", "age" : "18"}
data_2 = {"name" : "sds", "uid" : "6688"}
out_data = {**data_1, **data_2}
print(out_data)

>>> {'name': 'sds', 'age': '18', 'uid': '6688'}
```



#### 判断某对象是否为某些值

如果需要在if中将某对象与多个其他对象进行对比判断，你可能会进行如下定义

```python
data = "a"
if data == "a" or data == "b" or data == "c":
    print("HHH")

>>> HHH
```

简化修改为

```python
datas = ["a", "b", "c"]
data = "a"
if data in datas:
    print("HHH")

>>> HHH
```

#### 注意^和**的区别

按位异或  和   次幂

#### 写文件时请用with

当写入出错时会报错，文件会关闭

#### 数字中的下划线

`x=10000000`和`x=10_000_000`等价，但是后者明显更加清晰

#### 没穿衣服的元组

```python
x = 1,2
d1 = x[0]
d2 = x[1]
```

#### 使用isinstance代替==号检查类型

```python
type(name) == tuple

isinstance(name, tuple)
```

#### b,a  = a,b快速值交换