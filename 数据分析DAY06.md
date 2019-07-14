# 数据分析DAY06

## 符号数组

sign函数可以把样本数组的变成对应的符号数组，正数变为1，负数变为-1，0则变为0。

```python
ary = np.sign(源数组)
```

**净额成交量（OBV）**

成交量可以反映市场对某支股票的人气，而成交量是一只股票上涨的能量。一支股票的上涨往往需要较大的成交量。而下跌时则不然。

若相比上一天的收盘价上涨，则为正成交量；若相比上一天的收盘价下跌，则为负成交量。

绘制OBV柱状图

```python
dates, closing_prices, volumes = np.loadtxt(
    '../../data/bhp.csv', delimiter=',',
    usecols=(1, 6, 7), unpack=True,
    dtype='M8[D], f8, f8', converters={1: dmy2ymd})
diff_closing_prices = np.diff(closing_prices)
sign_closing_prices = np.sign(diff_closing_prices)
obvs = volumes[1:] * sign_closing_prices
mp.figure('On-Balance Volume', facecolor='lightgray')
mp.title('On-Balance Volume', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('OBV', fontsize=14)
ax = mp.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
dates = dates[1:].astype(md.datetime.datetime)
mp.bar(dates, obvs, 1.0, color='dodgerblue',
       edgecolor='white', label='OBV')
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

**数组处理函数**

```python
ary = np.piecewise(源数组, 条件序列, 取值序列)
```

针对源数组中的每一个元素，检测其是否符合条件序列中的每一个条件，符合哪个条件就用取值系列中与之对应的值，表示该元素，放到目标 数组中返回。

条件序列: [a < 0, a == 0, a > 0]

取值序列: [-1, 0, 1]     

```python
a = np.array([70, 80, 60, 30, 40])
d = np.piecewise(
    a, 
    [a < 60, a == 60, a > 60],
    [-1, 0, 1])
# d = [ 1  1  0 -1 -1]
```

## 矢量化

矢量化指的是用数组代替标量来操作数组里的每个元素。

numpy提供了vectorize函数，可以把处理标量的函数矢量化，返回的函数可以直接处理ndarray数组。

```python
import math as m
import numpy as np

def foo(x, y):
    return m.sqrt(x**2 + y**2)

x, y = 1, 4
print(foo(x, y))
X, Y = np.array([1, 2, 3]), np.array([4, 5, 6])
vectorized_foo = np.vectorize(foo)
print(vectorized_foo(X, Y))
print(np.vectorize(foo)(X, Y))
```

numpy还提供了frompyfuc函数，也可以完成与vectorize相同的功能：

```python
# 把foo转换成矢量函数，该矢量函数接收2个参数，返回一个结果 
fun = np.frompyfunc(foo, 2, 1)
fun(X, Y)
```

案例：定义一种买进卖出策略，通过历史数据判断这种策略是否值得实施。

```python
"""
demo03_vec.py  定义一种投资策略  判断是否可以实施
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt
import matplotlib.dates as md

def dmy2ymd(dmy):
	# 把日月年字符串转为年月日字符串
	dmy = str(dmy, encoding='utf-8')
	d = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
	ymd = d.strftime('%Y-%m-%d')
	return ymd


# 加载文件
dates, opening_prices, highest_prices, \
lowest_prices, closing_prices = \
    np.loadtxt('../da_data/aapl.csv',
	delimiter=',', usecols=(1, 3, 4, 5, 6),
	unpack=True, dtype='M8[D], f8, f8, f8, f8',
	converters={1:dmy2ymd})

# 绘制收盘价的折线图
mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=18)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
mp.grid(linestyle=':')
mp.tick_params(labelsize=10)
# 设置刻度定位器
ax = mp.gca()
#每周一一个主刻度
maloc = md.WeekdayLocator(byweekday=md.MO)
ax.xaxis.set_major_locator(maloc)
# 设置主刻度日期的格式
ax.xaxis.set_major_formatter(
	md.DateFormatter('%Y-%m-%d'))
#DayLocator:每天一个次刻度
ax.xaxis.set_minor_locator(md.DayLocator())
# 把dates的数据类型改为matplotlib的日期类型
dates = dates.astype(md.datetime.datetime)


def profit(oprice, hprice, lprice, cprice):
	# 定义买入卖出策略函数
	buy_price = oprice * 0.99
	if lprice <= buy_price <= hprice:
		return (cprice-buy_price)/buy_price
	else:
		return np.nan

# 计算使用该策略后，30天中每天的收益
profits=np.vectorize(profit)(opening_prices, 
	highest_prices, lowest_prices, 
	closing_prices)
print(profits)
#  获取profits中isnan的掩码数组
nan_mask = np.isnan(profits)
dates = dates[~nan_mask]
profits = profits[~nan_mask]
# 绘制收益线
mp.plot(dates, profits, 'o-', label='profit')
print(np.mean(profits))

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
```

## 矩阵

矩阵是numpy.matrix类类型的对象，该类继承自numpy.ndarray，任何针对多维数组的操作，对矩阵同样有效，但是作为子类矩阵又结合其自身的特点，做了必要的扩充，比如：乘法计算、求逆等。

**矩阵对象的创建**

```python
# 如果copy的值为True(缺省)，所得到的矩阵对象与参数中的源容器
# 独立两份数据
numpy.matrix(
    ary,		# 任何可被解释为矩阵的二维容器
  	copy=True	# 是否复制数据(缺省值为True，即复制数据)
)
```

```python
# 等价于：numpy.matrix(..., copy=False)
# 由该函数创建的矩阵对象与参数中的源容器一定共享数据，无法拥有独立的数据拷贝
numpy.mat(任何可被解释为矩阵的二维容器)
```

```python
# 该函数可以接受字符串形式的矩阵描述：
# 数据项通过空格分隔，数据行通过分号分隔。例如：'1 2 3; 4 5 6'
numpy.mat(拼块规则)
```

**矩阵的乘法运算**

```python
# 矩阵的乘法：乘积矩阵的第i行第j列的元素等于
# 被乘数矩阵的第i行与乘数矩阵的第j列的点积
#
#           1   2   6
#    X----> 3   5   7
#    |      4   8   9
#    |
# 1  2  6   31  60  74
# 3  5  7   46  87 116
# 4  8  9   64 120 161
e = np.mat('1 2 6; 3 5 7; 4 8 9')
print(e * e)
```

**矩阵的逆矩阵**

若两个矩阵A、B满足：AB = BA = E （E为单位矩阵），则成为A、B为逆矩阵。

```python
# 测试矩阵的逆
e = np.mat('10 3 6;3 50 7; 4 8 90')
# e = np.mat('1 2 3; 4 5 6; 7 8 9')
print(e)
print(e.I)
print(e * e.I)

# 广义逆矩阵 --矩阵求逆的过程推广到非方阵
e = np.mat('1 4 6; 4 6 7')
print(e)
print(e.I)
print(e * e.I)
```

ndarray提供了方法让多维数组替代矩阵的运算： 

```python
a = np.array([
    [1, 2, 6],
    [3, 5, 7],
    [4, 8, 9]])
# 点乘法求ndarray的点乘结果，与矩阵的乘法运算结果相同
k = a.dot(a)
print(k)
# linalg模块中的inv方法可以求取a的逆矩阵
l = np.linalg.inv(a)
print(l)
```

案例：假设一帮孩子和家长出去旅游，去程坐的是bus，小孩票价为3元，家长票价为3.2元，共花了118.4；回程坐的是Train，小孩票价为3.5元，家长票价为3.6元，共花了135.2。分别求小孩和家长的人数。使用矩阵求解。
$$
\left[ \begin{array}{ccc}
	3 & 3.2 \\
	3.5 & 3.6 \\
\end{array} \right]
\times
\left[ \begin{array}{ccc}
	x \\
    y \\
\end{array} \right]
=
\left[ \begin{array}{ccc}
	118.4 \\
	135.2 \\
\end{array} \right]
$$

```python
import numpy as np

prices = np.mat('3 3.2; 3.5 3.6')
totals = np.mat('118.4; 135.2')

persons = prices.I * totals
print(persons)
```

案例：斐波那契数列

1	1	 2	 3	5	8	13	21	34 ...

```python
X      1   1    1   1    1   1
       1   0    1   0    1   0
    --------------------------------
1  1   2   1    3   2    5   3
1  0   1   1    2   1    3   2
 F^1    F^2      F^3 	  F^4  ...  f^n
```

**代码**

```python
import numpy as np
n = 35

# 使用递归实现斐波那契数列
def fibo(n):
    return 1 if n < 3 else fibo(n - 1) + fibo(n - 2)
print(fibo(n))

# 使用矩阵实现斐波那契数列
print(int((np.mat('1. 1.; 1. 0.') ** (n - 1))[0, 0]))
```

## 通用函数

**数组的裁剪**

```python
# 将调用数组中小于和大于下限和上限的元素替换为下限和上限，返回裁剪后的数组，调用数组保持不变。
ndarray.clip(min=下限, max=上限)
```

**数组的压缩**

```python
# 返回由调用数组中满足条件的元素组成的新数组。
ndarray.compress(条件)
a.compress(np.all([a>3, a<5], axis=0))
```

**数组的累乘**

```python
# 返回调用数组中所有元素的乘积——累乘。
ndarray.prod()
# 返回调用数组中所有元素执行累乘的过程数组。
ndarray.cumprod()
```

案例：

```python
from __future__ import unicode_literals
import numpy as np
a = np.array([10, 20, 30, 40, 50])
print(a)
b = a.clip(min=15, max=45)
print(b)
c = a.compress((15 <= a) & (a <= 45))
print(c)
d = a.prod()
print(d)
e = a.cumprod()
print(e)

def jiecheng(n):
    return n if n == 1 else n * jiecheng(n - 1)

n = 5
print(jiecheng(n))
jc = 1
for i in range(2, n + 1):
    jc *= i
print(jc)
print(np.arange(2, n + 1).prod())
```



### 加法、乘法通用函数 

```python
np.add(a, a) 					# 两数组相加
np.add.reduce(a) 				# a数组元素累加和
np.add.accumulate(a) 			# 累加和过程
np.add.outer([10, 20, 30], a)	# 外和
np.prod(a)						# 累乘
np.cumprod(a)					# 累乘过程
np.outer([10, 20, 30], a)	# 外积
```

案例：

```python
"""
demo05_add.py  加法乘法通用函数
"""
import numpy as np

a = np.arange(1, 7)
print(a)

print(np.add(a, a))
print(np.add.reduce(a))
print(np.add.accumulate(a))
print(np.prod(a))
print(np.cumprod(a))

print(np.add.outer([10, 20, 30], a))
print(np.outer([10, 20, 30], a))
```

### 除法、取整通用函数

```python
np.true_divide(a, b) 	# a 真除 b  （对应位置相除）
np.divide(a, b) 		# a 真除 b
np.floor_divide(a, b)	# a 地板除 b（真除的结果向下取整）
np.ceil(a / b) 			# 向上取整
np.trunc(a / b)			# 截断取整
np.round(a / b)         # 四舍五入
```

案例：

```python
import numpy as np

a = np.array([20, 20, -20, -20])
b = np.array([3, -3, 6, -6])
# 真除
c = np.true_divide(a, b)
c = np.divide(a, b)
c = a / b
print('array:',c)
# 对ndarray做floor操作
d = np.floor(a / b)
print('floor_divide:',d)
# 对ndarray做ceil操作
e = np.ceil(a / b)
print('ceil ndarray:',e)
# 对ndarray做trunc操作
f = np.trunc(a / b)
print('trunc ndarray:',f)
# 对ndarray做around操作
g = np.around(a / b)
print('around ndarray:',g)
```

### 位运算通用函数

**位异或：**

```python
c = a ^ b
c = a.__xor__(b)
c = np.bitwise_xor(a, b)
```

二进制：

```
-8	1000    1000 0000 0000 0000 0000 0000 0000 0000 
-7	1001
-6	1010
-5	1011
-4	1100
-3	1101
-2	1110
-1	1111
0	0000    0000 0000 0000 0000 0000 0000 0000 0000 
1	0001    .....
2	0010
3	0011
4	0100
5	0101
6	0110
7	0111    0111 1111 1111 1111 1111 1111 1111 1111  
```

按位异或操作可以很方便的判断两个数据是否同号。

```
0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0
```

```python
a = np.array([0, -1, 2, -3, 4, -5])
b = np.array([0, 1, 2, 3, 4, 5])
print(a, b)
c = a ^ b
# c = a.__xor__(b)
# c = np.bitwise_xor(a, b)
print(np.where(c < 0)[0])
```

**位与：**

```python
e = a & b
e = a.__and__(b)
e = np.bitwise_and(a, b)
```

```
0 & 0 = 0 
0 & 1 = 0
1 & 0 = 0
1 & 1 = 1
```

利用位与运算计算某个数字是否是2的幂

```python
#  1 2^0 00001   0 00000
#  2 2^1 00010   1 00001
#  4 2^2 00100   3 00011
#  8 2^3 01000   7 00111
# 16 2^4 10000  15 01111
# ...

d = np.arange(1, 21)
print(d)
e = d & (d - 1)
e = d.__and__(d - 1)
e = np.bitwise_and(d, d - 1)
print(e)
```

**位或：**

```python
|
__or__
bitwise_or
```

0 | 0 = 0
0 | 1 = 1
1 | 0 = 1
1 | 1 = 1
**位反：**

```python
~
__not__
bitwise_not
```

~0 = 1
~1 = 0
**移位：**

```python
<<		__lshift__		left_shift
>>		__rshift__		right_shift
```

左移1位相当于乘2，右移1位相当于除2。

```python
d = np.arange(1, 21)
print(d)
# f = d << 1
# f = d.__lshift__(1)
f = np.left_shift(d, 1)
print(f)
```

### 三角函数通用函数

```python
numpy.sin()
```



**傅里叶定理**：法国科学家傅里叶提出，任何一条周期曲线，无论多么跳跃或不规则，都能表示成一组光滑正弦曲线叠加之和。

**合成方波**

一个方波由如下参数的正弦波叠加而成：
$$
y = 4\pi \times sin(x) \\
y = \frac{4\pi}{3} \times sin(3x) \\
...\\
...\\
y = \frac{4\pi}{2n-1} \times sin((2n-1)x)
$$
曲线叠加的越多，越接近方波。所以可以设计一个函数，接收曲线的数量n作为参数，返回一个矢量函数，该函数可以接收x坐标数组，返回n个正弦波叠加得到的y坐标数组。

```python
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.zeros(1000)
n = 1000
for i in range(1， n+1):
	y += 4 / ((2 * i - 1) * np.pi) * np.sin((2 * i - 1) * x)
mp.plot(x, y, label='n=1000')
mp.legend()
mp.show()
```

## 

