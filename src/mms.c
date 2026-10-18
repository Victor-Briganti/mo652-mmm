#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct timespec timespec;

//===------------------------------------------------------------------------===
// Matrix
//===------------------------------------------------------------------------===

static inline void matmul(const float *A, const float *B, float *restrict C,
                          size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Wrong number of arguments\n");
    fprintf(stderr, "Example Usage: %s <N> output.csv\n", argv[0]);
    return -1;
  }

  int64_t N = atoll(argv[1]);
  if (N < 0) {
    fprintf(stderr, "N should not be a negative number\n");
    return -1;
  }

  timespec before;
  timespec after;

  FILE *file = fopen(argv[2], "w");

  float *A = malloc(sizeof(float) * N * N);
  float *B = malloc(sizeof(float) * N * N);
  float *C = calloc(N * N, sizeof(float));

  matrix_fill(A, N);
  matrix_fill(B, N);

  clock_gettime(CLOCK_MONOTONIC, &before);
  matmul(A, B, C, N);
  clock_gettime(CLOCK_MONOTONIC, &after);

  matrix_print(C, N, file);
  fprintf(stdout, "1,1,%lux%lu,%ld\n", N, N, elapsed_time(before, after));
  return 0;
}