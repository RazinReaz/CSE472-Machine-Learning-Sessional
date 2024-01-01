import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)
print(x.shape)

print(x.reshape(3, 2))
print(x.reshape(6, 1))
print(x.flatten())