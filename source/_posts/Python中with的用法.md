---
title: Python中with的用法
top: false
cover: false
toc: true
mathjax: true
date: 2019-11-28 00:14:20
password:
summary:
tags: 
- python
categories: Python
---



# 一、什么是with语句

​	对于系统资源如文件、数据库连接、socket 而言，应用程序打开这些资源并执行完业务逻辑之后，必须做的一件事就是要关闭（断开）该资源。

```python
# 操作文件的方式

# 普通方式操作文件
f = open('test.txt', 'w')
f.write('test')
f.close()
# 普通方式操作文件的问题
# 1、忘记关闭文件
# 2、程序执行的过程中发生了异常导致关闭文件的代码没有被执行

# 可以使用try...finally的方式解决上述问题
try:
    f = open('test.txt', 'w')
	f.write('test')
finally:
	f.close()
    
# Python提供了一种更简单的解决方案:with语句
# 不论执行过程中是否发生异常，文件都会关闭
with open('test.txt', 'w') as f:
    f.write('test')

```

**with 语句的语法格式**:

```shell
with 表达式 [as目标]:	
	代码块		
```

​								

#  二、with语句的原理

1. 上下文管理器

   上下文管理器是一个实现了上下文协议的对象，即在对象中定义了`__enter__`和`__exit__`方法。上下文管理器定义运行时需要建立的上下文，处理程序的进入和退出。

   ```python
   # open函数返回的就是一个上下文管理器对象
   f.__enter__   # <function TextIOWrapper.__enter__>
   f.__exit__    # <function TextIOWrapper.__exit__>
   ```

   `__enter__(self)`:该方法只接收一个self参数。当对象返回时该方法立即执行，并返回当前对象或者与运行时上下文相关的其他对象。如果有as变量（as子句是可选项），返回值将被赋给as后面的变量上。

   `__exit__(self, exception_type, exception_value, traceback)`:退出运行时上下文，并返回一个布尔值标示是否有需要处理的异常。如果在执行with语句体时发生异常，那退出时参数会包括异常类型、异常值、异常追踪信息，否则，3个参数都是None。返回True异常被处理，返回其他任何值，异常都会抛出。

   

2. 自定义上下文管理器

   任何实现了上下文管理器协议的对象都可以称为一个上下文管理器。

   ```python
   # 实现一个简单的open()上下文管理
   class myopen:
       def __init__(self, name, mode='r'):
           self.file = open(name, mode)          
       def __enter__(self):
           return self.file
       def __exit__(self, exception_type, exception_value, traceback):
           self.file.close()
           return False
               
   with myopen('test.txt') as f:
        print(f.read())  # test
   f.closed  # True
   
   # 同时打开多个文件
   with myopen('test1.txt', 'w') as f1, myopen('test2.txt', 'w') as f2, myopen('test3.txt', 'w') as f3:
       f1.write('test1')
       f2.write('test2')
       f3.write('test3')
   ```



# 三、contextlib

​	为了更好的辅助上下文管理，python提供了`contextlib`模块，该模块通过`Generator`实现，其中的`contextmanager`作为装饰器来提供了一种针对函数级别的上下文管理机制，可以直接使用与函数/对象而不用关心`__enter__`和`__exit__`方法的具体实现。

1. `contextmanager`

   ```python
   from contextlib import contextmanager
   
   @contextmanager
   def myopen(name, model='r'):
       try:
           f = open(name, model)
           yield f
       finally:
           f.close()
           
   with myopen('test.txt') as f:
        print(f.read())  # test
   f.closed  # True
   ```



2. `closing`

   文件类是支持上下文管理协议的，可以直接用with语句，还有一些对象并不支持该协议，但使用的时候又要确保正常退出，这时就可以使用closing创建上下文管理器。

```python
from urllib import request
from contextlib import closing, contextmanager
with closing(request.urlopen('https://www.baidu.com/')) as f:
	data = f.read()
	print('Status:', f.status, f.reason)
	print('Data:', data)

# 等价于:
@contextmanager
def closing(f):
	try:
		yield f
	finally:
		f.close()
```

   