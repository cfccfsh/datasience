"""
demo08_fly.py  验证傅里叶定理，合成方波
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y1 = 4*np.pi * np.sin(x)
y2 = 4/3 * np.pi * np.sin(3*x)

n = 1000
y = np.zeros(x.size)
for i in range(1, n+1):
	y += 4*np.pi/(2*i-1) * np.sin((2*i-1)*x)

mp.grid(linestyle=':')
mp.plot(x, y1, label='y1', alpha=0.3)
mp.plot(x, y2, label='y2', alpha=0.3)
mp.plot(x, y1+y2, label='y1+y2', alpha=0.3)
mp.plot(x, y, label='y')

mp.legend()
mp.show()