#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  // Matrix dimensions
  // A is m x k
  // B is k x n
  // C is m x n
  int m = 3;
  int n = 3;
  int k = 3;

  // Scalars for the formula: C = alpha * A * B + beta * C
  double alpha = 1.0;
  double beta = 0.0;

  // Allocate memory for 1D arrays representing 2D matrices
  double *A = (double *)malloc(m * k * sizeof(double));
  double *B = (double *)malloc(k * n * sizeof(double));
  double *C = (double *)malloc(m * n * sizeof(double));

  // Initialize A with values 1.0 to 9.0
  for (int i = 0; i < m * k; i++) {
    A[i] = i + 1.0;
  }

  // Initialize B as an Identity matrix (1.0 on diagonal, 0.0 elsewhere)
  for (int i = 0; i < k * n; i++) {
    B[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
  }

  // Initialize C with zeros
  for (int i = 0; i < m * n; i++) {
    C[i] = 0.0;
  }

  /*
   * Perform matrix multiplication
   * cblas_dgemm computes: C = alpha * A * B + beta * C
   */
  cblas_dgemm(CblasRowMajor, // We are using row-major C arrays
              CblasNoTrans,  // Do not transpose matrix A
              CblasNoTrans,  // Do not transpose matrix B
              m, n, k,       // Dimensions m, n, k
              alpha,         // Alpha scalar
              A, k,          // Matrix A and its leading dimension
              B, n,          // Matrix B and its leading dimension
              beta,          // Beta scalar
              C, n);         // Matrix C and its leading dimension

  // Print the resulting matrix C
  printf("Resulting Matrix C (A * Identity):\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%6.1f ", C[i * n + j]);
    }
    printf("\n");
  }

  // Clean up memory
  free(A);
  free(B);
  free(C);

  return 0;
}