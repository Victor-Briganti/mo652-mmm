#include "summa.h"
#include "utils.h"

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  const SummaVariant variant = {
      .name = "mpi_blas",
      .gemm = cgemm,
      .usesTile = false,
      .threads = openblas_get_num_threads(),
  };
  return summa(argc, argv, &variant);
}
