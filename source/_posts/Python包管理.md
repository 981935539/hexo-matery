---
title: Python包管理
top: false
cover: false
toc: true
mathjax: true
date: 2019-11-28 00:08:24
password:
summary:
tags: 
- python
categories: Python
---




# 一、什么是包？

简单来说包即是目录，但和普通目录不同，它除了包含python文件以外，还包含一个`__init__.py`文件，同时它允许嵌套。

包结构如下：

```python
package/__init__.py
		module1.py
    		class C1:pass
    	module2.py
        	class C2:pass
        subpackage/__init__.py
        		   module1.py        			
               	   module2.py              	
                   module3.py

main.py
                    
import package
import package.module1
import package.subpackage
import package.subpackage.module1

from package import module1
from package import subpackage
from package.subpackage import module1
# from package import module3
```



# 二、`__init__.py`的作用

1. 区别包和普通目录

2. 可以使模块中的对象变成包可见

   例如：要导入package包下module1中的类Test, 当`__init__.py`文件为空的时候需要使用完整的导入路径:`from package.module import Test`, 但如果在`__init__.py`中添加`from module1 import Test`语句，就可以直接使用`from package import Test`来导入类Test。

3. 通过在该文件中定义`__all__`变量，控制需要导入的子包或模块。

   


# 三、`__all__`的作用

​	`__all__`只控制`from xxx import *`的行为, 不影响`import` 和 `from xxx import xxxx`的行为

1. 在`__init__.py`文件中添加：

   ​					`__all__ = ['module1', 'subpackage']`

   `__init__.py`不使用`__all__`属性，不能通过`from package import * `导入

   `__init__.py`使用`__all__`属性，`from package import *`只能导入`__all__`列表中的成员，但可以通过

   `import package.module2`和`from package import module2`导入

2. 在普通`*.py`文件中添加：`__all__`

   ```python
   # from xxx import * 这种方式只能导入公有的属性，方法或类【无法导入以单下划线开头（protected）或以双下划线开头(private)的属性，方法或类】
   
   # from xxx import aa, bb 可以导入public,protected,private
   # import xxx   xxx.__func  可以访问public,protected,private
   ```

   模块中不使用`__all__`属性，可以导入模块内的所有公有属性，函数和类 。

   模块中使用`__all__`属性，只能导入`__all__`中定义的属性，函数和类(包括私有属性和保护属性)。



# 四、`from ... import ...`的问题

1. 命名空间的冲突

   ```python
   # module1.py
   def add()
   	print("add in module1")
   # module1.py
   def add()
   	print("add in module2")
       
   # main.py
   from package.module1 import add
   from package.module2 import add
   
   # 最近导入的add,会覆盖先导入的add
   ```

   

2. 循环嵌套导入的问题

   ```python
   # module1.py
   from module2 import g
   def x()
   	pass
   # module2.py
   from module1 import x
   def g()
   	pass
   
   # 会抛出一个ImportError: cannot import name 'g'异常，解决方法直接使用import 语句
   ```



