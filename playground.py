import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1], [0]])
c = a[np.where(b == 2)[0], :]
print(c)