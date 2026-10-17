import sys

import numpy as np


def matmul(size):
    A = [x for x in range(size * size)]
    A = np.array(A).reshape(size, size)
    B = A.copy()
    return A @ B


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Wrong number of arguments")
        print(f"Usage: {sys.argv[0]} <matrix_size> <compare_file>")

    N = int(sys.argv[1])
    C = matmul(N)

    data = np.loadtxt(sys.argv[2], delimiter=",")

    if not np.allclose(data, C):
        print("Generated files are not equal")
        sys.exit(-1)
