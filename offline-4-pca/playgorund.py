import numpy as np


x = np.ones(3)
y = np.array([[1, 2, 3, 10],
              [8, 4, 6, 10],
              [7, 5, 9, 10]])
print(y.shape)
print(y[:0])
print(y[:1])
print(y[:2])
print(y[:3])

print(y[:0].flatten())
print(y[:1].flatten())
print(y[:2].flatten())
print(y[:3].flatten())