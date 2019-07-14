import math as m
import numpy as np
import matplotlib.pyplot as  mp



def foo(x, y):
    return m.sqrt(x ** 2 + y ** 2)


x, y = 3, 4
print(foo(x, y))
X, Y = np.array([3, 4, 5]), np.array([4, 5, 6])
vectorized_foo = np.vectorize(foo)
print(vectorized_foo(X, Y))
print(np.vectorize(foo)(X, Y))

func = np.frompyfunc(foo, 2, 1)
a = func(3, 4)
print(a)

b = np.arange(1, 9)
print(b)
# c = np.ndarray.clip(b,min=3,max=6)
# print(c)

d = b.compress(np.all([b > 3, b < 6], axis=0))
print(d)

e = b.prod()
print(e)
f = b.cumprod()
print(f)

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
print("*" * 50)

import numpy as np

a = np.array([20, 20, -20, -20, 7])
b = np.array([3, -3, 6, -6, 2])
# 真除
c = np.true_divide(a, b)
c = np.divide(a, b)
c = a / b
print('array:', c)
# 对ndarray做floor操作
d = np.floor(a / b)
print('floor_divide:', d)
# 对ndarray做ceil操作
e = np.ceil(a / b)
print('ceil ndarray:', e)
# 对ndarray做trunc操作
f = np.trunc(a / b)
print('trunc ndarray:', f)
# 对ndarray做around操作
g = np.around(a / b)
print('around ndarray:', g)
print("-" * 50)
a = np.array([0, -1, 2, -3, 4, -5])
b = np.array([0, 1, 2, 3, 4, 5])
print(a, b)
c = a ^ b
print(c)
c = a.__xor__(b)
print(c)
c = np.bitwise_xor(a, b)
print(c)
print(np.where(c < 0))
d = np.arange(1, 21)
print(d)
f = d << 1
# f = d.__lshift__(1)
# f = np.left_shift(d, 1)
print(f)
a = np.array([2,3,5])
print(a+2)

x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.zeros(1000)
n = 1000
y1 = 4 / ((2 * 1 - 1) * np.pi) * np.sin((2 * 1 - 1) * x)
y2 = 4 / ((2 * 2 - 1) * np.pi) * np.sin((2 * 2 - 1) * x)
for i in range(1,3):
    a = np.sin((2 * i - 1) * x)
    print(a.size)
    y += 4 / ((2 * i - 1) * np.pi) * np.sin((2 * i - 1) * x)
print("y",y.size)
mp.plot(x, y1, label='n=1')
mp.plot(x, y2, label='n=2')
mp.plot(x, y, label='6')

mp.legend()
mp.show()
