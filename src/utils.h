#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

//===------------------------------------------------------------------------===
// Structs
//===------------------------------------------------------------------------===

typedef struct TimeSpend {
  struct timespec before;
  struct timespec after;
} TimeSpend;

//===------------------------------------------------------------------------===
// Time Utils
//===------------------------------------------------------------------------===

static inline void start_time(TimeSpend *timer) {
  clock_gettime(CLOCK_MONOTONIC, &timer->before);
}

static inline void finish_time(TimeSpend *timer) {
  clock_gettime(CLOCK_MONOTONIC, &timer->after);
}

static inline int64_t elapsed_time(TimeSpend *timer) {
  int64_t seconds =
      (int64_t)(timer->after.tv_sec - timer->before.tv_sec) * 1000000000LL;
  int64_t nanosec = timer->after.tv_nsec - timer->before.tv_nsec;
  return seconds + nanosec;
}

//===------------------------------------------------------------------------===
// Matrix Utils
//===------------------------------------------------------------------------===

static inline void matrix_print(const float *A, size_t size, FILE *file) {
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

static inline void matrix_fill(float *M, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      // Generates a pseudo random float
      M[i * N + j] = (float)((int)((i * 7 + j * 13) % 9 - 4));
    }
  }
}

//===------------------------------------------------------------------------===
// Other Utils
//===------------------------------------------------------------------------===

static inline int gcd(int a, int b) {
  while (b != 0) {
    int tmp = a % b;
    a = b;
    b = tmp;
  }

  return a;
}

#endif // MAT_UTILS_H