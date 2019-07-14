"""
demo02_vec.py  矢量化
"""
import numpy as np
import math as m

def foo(x, y):
	return m.sqrt(x**2 + y**2)

a = 3
b = 4
print(foo(a, b))

# 把foo函数矢量化，使之可以处理矢量数据
foovec = np.vectorize(foo)
a = np.array([3,4,5,6])
b = np.array([4,5,6,7])
print(foovec(a, b))

func = np.frompyfunc(foo, 2, 1)
print(func(a, b))
