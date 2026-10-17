#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct timespec timespec;

void mat_print(float *A, size_t size, FILE *file) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      fprintf(file, "%f", A[i * size + j]);

      if (j < size - 1) {
        fprintf(file, ",");
      }
    }
    fprintf(file, "\n");
  }
}

void mat_alloc(float *A, size_t size) {
  float x = 0.0;
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      A[i * size + j] = x;
      x++;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Wrong number of arguments\n");
    fprintf(stderr, "Usage: %s <matrix_size> <output>\n", argv[0]);
  }

  uint64_t N = atoll(argv[1]);

  timespec before;
  timespec after;

  FILE *file = fopen(argv[2], "w");

  float *A = malloc(sizeof(float) * N * N);
  float *B = malloc(sizeof(float) * N * N);
  float *C = calloc(N * N, sizeof(float));

  mat_alloc(A, N);
  mat_alloc(B, N);

  clock_gettime(CLOCK_MONOTONIC, &before);
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &after);

  mat_print(C, N, file);

  fprintf(stdout, "%lux%lu,%ld\n", N, N, after.tv_nsec - before.tv_nsec);
  return 0;
}