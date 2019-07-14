"""
demo07_bitwise.py  位运算符
"""
import numpy as np

a = np.array([0, -1, 2, -3, 4, -5])
b = np.array([0, 1, 2, 3, 4, 5])

print(a^b)
c = a^b
# 找到符合条件的元素索引
print(np.where(c<0))