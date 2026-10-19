#include "summa.h"

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("%d of %d", rank, size);

  MPI_Finalize();
  return 0; 
}
