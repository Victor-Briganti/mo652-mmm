#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

inline int64_t elapsed_time(struct timespec before, struct timespec after) {
  int64_t seconds = (int64_t)(after.tv_sec - before.tv_sec) * 1000000000LL;
  int64_t nanosec = after.tv_nsec - before.tv_nsec;
  return seconds + nanosec;
}

inline void matrix_print(const float *A, size_t size, FILE *file) {
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

inline void matrix_fill(float *M, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      // Generates a pseudo random float
      M[i * N + j] = (float)((int)((i * 7 + j * 13) % 9 - 4));
    }
  }
}

#endif // MAT_UTILS_H