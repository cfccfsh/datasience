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
