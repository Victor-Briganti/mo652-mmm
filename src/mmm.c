#include "summa.h"

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  const SummaVariant variant = {
      .name = "mpi",
      .gemm = gemm,
      .usesTile = true,
      .threads = 1,
  };
  return summa(argc, argv, &variant);
}
