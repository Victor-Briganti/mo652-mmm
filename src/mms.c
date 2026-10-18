#include "utils.h"
#include <stdlib.h>

typedef struct timespec timespec;

//===------------------------------------------------------------------------===
// Matrix
//===------------------------------------------------------------------------===

static inline void matmul(const float *A, const float *B, float *restrict C,
                          size_t N, size_t T) {
  for (size_t ii = 0; ii < N; ii += T) {
    for (size_t jj = 0; jj < N; jj += T) {
      for (size_t kk = 0; kk < N; kk += T) {
        size_t ie = MIN(ii + T, N);
        size_t je = MIN(jj + T, N);
        size_t ke = MIN(kk + T, N);

        for (size_t i = ii; i < ie; i++) {
          for (size_t k = kk; k < ke; k++) {
            float a = A[i * N + k];
            for (size_t j = jj; j < je; j++) {
              C[i * N + j] += a * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Wrong number of arguments\n");
    fprintf(stderr, "Example Usage: %s <N> <T> output.csv\n", argv[0]);
    return -1;
  }

  int64_t N = atoll(argv[1]);
  int64_t T = atoll(argv[2]);
  if (N < 0 || T < 0) {
    fprintf(stderr, "N should not be a negative number\n");
    return -1;
  }

  if (T >= N) {
    fprintf(stderr, "T needs to be a less than N\n");
    return -1;
  }

  TimeSpend timer;
  FILE *file = fopen(argv[3], "w");
  float *A = malloc(sizeof(float) * N * N);
  float *B = malloc(sizeof(float) * N * N);
  float *C = calloc(N * N, sizeof(float));
  matrix_fill(A, N);
  matrix_fill(B, N);

  start_time(&timer);
  matmul(A, B, C, N, T);
  finish_time(&timer);

  matrix_print(C, N, file);
  fprintf(stdout, "1,1,%lux%lu,%ld\n", N, N, elapsed_time(&timer));
  return 0;
}