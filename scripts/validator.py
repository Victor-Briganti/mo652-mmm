import sys

import numpy as np


def generate_matrix(size):
    i, j = np.indices((size, size), dtype=np.int64)
    return ((i * 7 + j * 13) % 9 - 4).astype(np.float32)


def matmul(size):
    A = generate_matrix(size)
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
