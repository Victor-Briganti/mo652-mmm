#include "utils.h"
#include <stdlib.h>

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  Parameters opts;
  if (!parse_options(argc, argv, &opts)) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  size_t N = (size_t)opts.N;
  float *A = malloc(sizeof(float) * N * N);
  float *B = malloc(sizeof(float) * N * N);
  float *C = calloc(N * N, sizeof(float));
  matrix_fill(A, N);
  matrix_fill(B, N);

  TimeSpend timer;
  start_time(&timer);
  sgemm(A, B, C, N);
  finish_time(&timer);

  if (opts.output) {
    FILE *file = fopen(opts.output, "w");
    if (file == NULL) {
      fprintf(stderr, "Could not open file %s\n", opts.output);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      return EXIT_FAILURE;
    }

    matrix_print(C, opts.N, file);
  }

  print_result("serial", 1, 1, 1, 1, opts.N, 0,  elapsed_time(&timer), NULL);

  free(A);
  free(B);
  free(C);
  return EXIT_SUCCESS;
}
