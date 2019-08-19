import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
import time

a = 1
b = 2
c = 1

w_sigma = 1


def func(_x, p):
    _a, _b, _c = p
    return np.exp(_a * _x ** 2 + _b * _x + _c)


def residuals(p, y, x):
    return y - func(x, p)


x_data = np.array(list(range(100))) / 100
y_data = np.exp(a * x_data ** 2 + b * x_data + c) +\
         np.random.normal(scale=w_sigma, size=x_data.shape)
tic = time.time()
plsq = sco.leastsq(residuals, np.array([0, 0, 0]), args=(y_data, x_data))
print('用时：', time.time() - tic, 's')
print('拟合结果：', plsq[0])
plt.ion()
plt.scatter(x_data, y_data, c='k')
plt.plot(x_data, func(x_data, plsq[0]), c='r')
plt.ioff()
plt.waitforbuttonpress()

