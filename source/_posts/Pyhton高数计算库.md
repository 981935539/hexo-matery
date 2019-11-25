---
title: Pyhton高数计算库
top: false
cover: false
toc: true
mathjax: true
date: 2019-11-25 15:57:29
password:
summary:
tags: 
- python
- math
- sympy
categories: Python
---



# math数学库

```python
# 导入math库
import math

# 常用数学常量
math.pi		# π
math.e
math.inf	# ∞
math.nan	# not a num

# 指数/对数/开平方
math.exp(a) 	# math.e**a
math.log(a)		# 自然底数 math.e
math.log(a, b)	# 以b为底，b**x = a
math.sqrt(a)	# 开平方

# 近似值
math.ceil(4.1)	# roud up to 5
math.floor(4.9) # roud up to 4

# 阶乘
math.factorial(a) # a!

# 最大公约数
math.gcd(35, 49)  # 7

# 三角函数
math.sin(math.pi/2)		# 1.0
math.cos()
math.tan()
math.asin(1)	# 1.5707963267948966
math.acos()
math.atan()

# 弧度角度转换
math.degrees()	# 弧度转角度
math.radians()	# 角度转弧度

```





# sympy代数运算库

```python
# 导入库
from sympy import *

# 有理数
Rational(1, 3)	# 1/3

# 特殊无理数
pi	# math.pi
E	# math.e
oo	# math.inf

# jupyter pretty print
init_printing(pretty_print=True)  # Pretty printing mode
N(pi) = pi.eval()	# 3.15..默认取前15位
# .n() and N() are equivalent to .evalf();

# 代数运算 用符号代替数进行运算
x = Symbol('x')	# 声明一个代数符号
x,y = symbols('x y') # 一次声明的多个代数符号
(x+y)**2  # (𝑥+𝑦)2

# 展开和分解
# 展开多项式
expand((x+y)**2)	# 𝑥2+2𝑥𝑦+𝑦2

# 展开三角函数
expand(cos(x+y)**2, trig=True)  # sin2(𝑥)sin2(𝑦)−2sin(𝑥)sin(𝑦)cos(𝑥)cos(𝑦)+cos2(𝑥)cos2(𝑦)

# 化简
simplify((x+x*y)/x)  # 1+y
```

## 累加运算

$$
\sum_{x=1}^{10} {\frac {1}{x^2 + 2x}}
$$

```python
expr = Sum(1/(x**2 + 2*x), (x, 1, 10))
expr # 上面公式
expr.evalf() # 求值 0.662878787878788
expr.doit()  # 175/264
```



## 累积运算

$$
\prod_{x=1}^{10} {\frac {1}{x^2 + 2x}}
$$

```python
expr = Product(1/(x**2 + 2*x), (x, 1, 10))
expr
expr.doit()	# 1/869100503040000
```



## 极限

$$
\lim_{n \to +\infty} \frac{1}{n(n+1)} \quad 
$$

```python
n = Symbol('n')
expr = limit(1/(n*(n+1)), n, oo)
expr	# 0

# 左极限和有极限
limit(1/x, x, 0, dir='+')
limit(1/x, x, 0, dir='-')
```



## 导数

```python
diff(x**2, x)	# 2x
diff(sin(2*x), x)	# 2cos(2𝑥)
diff(sin(x**2+2*x),x) # diff(E**x*(x + sin(x)), x)

# 高阶导数
# 二阶导数
diff(sin(2*x), x, 2)  # −4sin(2𝑥)
# 三阶导数
diff(sin(2*x), x, 3)	# −8cos(2𝑥)
```



## 积分

不指定区间
$$
\int_{-\infty}^\infty {x^2} \,{\rm dx}
$$

```python
integrate(2*x, x) # 𝑥2
integrate(sin(x), x) # −cos(𝑥)
```



指定区间[a, b]
$$
\int_a^b {x^2} \,{\rm dx}
$$

```python
integrate(2*x, (x, 0, 1)) # 1
integrate(cos(x), (x, -pi/2, pi/2)) # 2
```



## 解方程

```python
# 解一元方程
solve(x**2-3*x+2, x)	# [1, 2]

# 解二元方程
solve([x+5*y-2, -3*x+6*y-15], [x, y]) #{x:-3, y:1}
```



## 代数运算

```python
expr = x**2 + 2*x + 1
# 令x = 2
expr.subs(x, 2)	# 9b

# 令x=y+1
expr.subs(x, y+1)	# 2𝑦+(𝑦+1)2+3

# 多元函数代数
expr = x**3 + 4*x*y -z
expr.subs([(x,1), (y, 1), (z, 0)]) # 5

# 使用字符串
expr = sympify("x*2 + 4*x*y")
expr.subs([(x, 1), (y, 1)])	# 6
```



## 概率论

```python
from sympy import stats

#创建一个6个面的筛子
X = stats.Die('X', 6)
# 查看某个面出现的概率
stats.density(X).dict	# {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}

# 随机丢一次筛子
stats.sample(X)	# 4

# 	硬币
C = stats.Coin('C')	
stats.density(C).dict	# {H: 1/2, T: 1/2}

# 正态分布
Z = stats.Normal('Z', 0, 1)
# Z>1的概率
stats.P(Z > 1).evalf() # 0.158655253931457
```

