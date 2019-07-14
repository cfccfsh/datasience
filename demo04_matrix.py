"""
demo04_matrix.py  矩阵
"""
import numpy as np

data = [[1,2,3], [4,5,6]]
m = np.matrix(data, copy=True)
print(m, type(m))
data[0][0] = 999
print(m, type(m))

# 矩阵创建 2
m = np.mat(data)
print(m, type(m))

# 矩阵创建 3
m = np.mat('1 2 3; 4 5 6')
print(m, type(m))

# 测试矩阵的乘法
print(m * m.T)

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

# 解方程
A = np.mat('3  3.2; 3.5  3.6')
B = np.mat('118.4; 135.2')
# 基于最小二乘法 求得误差最小的最优结果
X = np.linalg.lstsq(A, B)[0] 
print(X)
# 基于矩阵算法求解方程组
X = A.I * B
print(X)
# 解方程组
X = np.linalg.solve(A, B)
print(X)


# 斐波那契数列
F = np.mat('1 1; 1 0')
print(F**10)