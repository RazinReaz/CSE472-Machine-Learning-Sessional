import numpy as np
import random
import math

def take_integer_input(message):
    while True:
        try:
            user_input = int(input(message))
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue
        else:
            if user_input <= 0:
                print("Invalid input. Please enter a positive integer.")
                continue
            return user_input


def elementary_row_operation(A):
    n = A.shape[0]
    r_lim = 3
    r = random.randint(-r_lim, r_lim)
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    while i == j:
        j = random.randint(0, n-1)
    A[i] += r*A[j]
    return A

def generate_symmetric_invertible_matrix(n):
    A = np.identity(n)
    r = random.randint(5, 10)
    for i in range(r):
        A = elementary_row_operation(A)
    return A.dot(A.T)

def is_symmetric(A):
    return np.allclose(A, A.T)

def test_invertibility_algorithm(n):
    count = 0
    m = 10000
    for i in range(m):
        A = generate_symmetric_invertible_matrix(n)
        if np.linalg.det(A) == 0 or not is_symmetric(A):
            count += 1
        print(f"\r: {i + 1}/{m}", end='', flush=True)
    print(f'\nNumber of peradayok matrices: {count}')
    if count == 0:
        print("All matrices generated are invertible and symmetric.")
    else:
        print("Not all matrices generated are invertible and symmetric.")


if __name__ == "__main__":
    n = take_integer_input("Enter the dimension of the square matrix: ")
    # test_invertibility_algorithm(n)
    A = generate_symmetric_invertible_matrix(n)

    # eigen decomposition
    eigen_values, eigen_vectors = np.linalg.eigh(A)
    X = eigen_vectors
    L = np.diag(eigen_values)
    X_inv = np.linalg.inv(X)
    result = X.dot(L).dot(X_inv)
    
    # printing
    print(f'Generated matrix:\n {A}')
    print(f'Eigen values:\n{eigen_values}')
    print(f'Eigen vectors:\n {eigen_vectors}')

    print(f'lambda:\n{L}')
    print(f'X:\n{X}')
    print(f'X_inv:\n{X_inv}')

    print(f'X*L*X_inv:\n{result}')
    print(f'A:\n{A}')

    if (np.allclose(A, result)):
        print("A == X*L*X_inv")
        print("Eigen decomposition successful.")
    else:
        print("Eigen decomposition failed.")
    