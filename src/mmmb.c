#include "summa.h"
#include "utils.h"

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  printf("%d\n", openblas_get_num_threads());
  return 0;
}
