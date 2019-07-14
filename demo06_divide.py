"""
demo06_divide.py   除法相关通用函数
"""
import numpy as np

a = np.array([20, 20, -20, -20])
b = np.array([3, -3, 6, -6])
print(a/b)
print(np.floor_divide(a, b))
print(np.ceil(a/b))
print(np.trunc(a/b))
print(np.round(a/b))
