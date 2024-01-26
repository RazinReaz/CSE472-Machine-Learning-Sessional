import numpy as np


A = np.array([[1, 2], [4, 5], [7, 8]])
n = np.array([[2],
              [3],
              [4]])

x = np.array([  [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10]])
mean = np.array([[2, 3],
                [4, 5],
                [6, 7]])

diff = x - mean[0]

print(diff)
print(np.outer(diff, diff))

