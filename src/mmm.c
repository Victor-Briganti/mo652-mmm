#include "utils.h"
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROOT (0)

static void matmul(int m, int n, int kd, const float *A, int lda,
                   const float *B, int ldb, float *restrict C, int ldc, int T) {
  for (int ii = 0; ii < m; ii += T)
    for (int jj = 0; jj < n; jj += T)
      for (int kk = 0; kk < kd; kk += T) {
        int ie = MIN(ii + T, m);
        int je = MIN(jj + T, n);
        int ke = MIN(kk + T, kd);
        for (int i = ii; i < ie; i++)
          for (int k = kk; k < ke; k++) {
            float a = A[(size_t)i * lda + k];
            for (int j = jj; j < je; j++)
              C[(size_t)i * ldc + j] += a * B[(size_t)k * ldb + j];
          }
      }
}

static void packblk(const float *full, float *packed, int N, int pr, int pc) {
  int mb = N / pr;
  int nb = N / pc;

  size_t blockSz = (size_t)mb * nb;
  size_t rowBytes = (size_t)nb * sizeof(float);

  for (int r = 0; r < pr * pc; r++) {
    int gridRow = r / pc;
    int gridCol = r % pc;

    size_t firstRow = (size_t)gridRow * mb;
    size_t firstCol = (size_t)gridCol * nb;

    float *block = &packed[r * blockSz];

    for (int i = 0; i < mb; i++) {
      const float *srcRow = &full[(firstRow + i) * N + firstCol];
      float *dstRow = &block[(size_t)i * nb];
      memcpy(dstRow, srcRow, rowBytes);
    }
  }
}

static void unpackblk(const float *packed, float *full, int N, int pr, int pc) {
  int mb = N / pr;
  int nb = N / pc;

  size_t blockSz = (size_t)mb * nb;
  size_t rowBytes = (size_t)nb * sizeof(float);

  for (int r = 0; r < pr * pc; r++) {
    int gridRow = r / pc;
    int gridCol = r % pc;

    size_t firstRow = (size_t)gridRow * mb;
    size_t firstCol = (size_t)gridCol * nb;

    const float *block = &packed[r * blockSz];

    for (int i = 0; i < mb; i++) {
      const float *srcRow = &block[(size_t)i * nb];
      float *dstRow = &full[(firstRow + i) * N + firstCol];
      memcpy(dstRow, srcRow, rowBytes);
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);
  int prows = dims[0];
  int pcols = dims[1];

  /// ===============
  /// Initialization
  /// ===============

  int params[2] = {
      0, /* N */
      0  /* T */
  };
  if (rank == ROOT) {
    if (argc < 4) {
      fprintf(stderr, "Wrong number of arguments");
      fprintf(stderr, "Usage: %s <N> <T> output.csv \n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    long N = atol(argv[1]);
    long T = atol(argv[2]);

    if (N <= 0 || T <= 0) {
      fprintf(stderr, "N and T must be positive integers\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    params[0] = N;
    params[1] = T;
  }
  MPI_Bcast(params, 2, MPI_INT, ROOT, MPI_COMM_WORLD);

  int N = params[0];
  int T = params[1];

  // Local block rows and columns
  int mb = N / prows;
  int nb = N / pcols;

  // Elements per local block
  int blk = mb * nb;

  // Panel width
  int panel = gcd(mb, nb);

  // Number of panels along the width of the matrix
  int steps = N / panel;

  /// ===============
  /// Grid Building
  /// ===============

  int periods[2] = {0, 0};
  int coords[2] = {0, 0};
  MPI_Comm grid, rowComm, colComm;

  // Builds a new communicator (grid) with a 2D layout.
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);

  // Gets the coords of this node in this new system
  MPI_Cart_coords(grid, rank, 2, coords);
  int myrow = coords[0];
  int mycol = coords[1];

  // Define the communication channel for the row and the column
  int rowDims[2] = {0, 1};
  int colDims[2] = {1, 0};
  MPI_Cart_sub(grid, rowDims, &rowComm);
  MPI_Cart_sub(grid, colDims, &colComm);

  /// ===============
  /// Allocation
  /// ===============

  float *Al = malloc(sizeof(float) * blk);
  float *Bl = malloc(sizeof(float) * blk);
  float *Ap = malloc(sizeof(float) * mb * panel);
  float *Bp = malloc(sizeof(float) * nb * panel);
  float *Cl = calloc(blk, sizeof(float));

  // Root holds the full matrices and one block-ordered staging buffer
  float *A = NULL;
  float *B = NULL;
  float *C = NULL;
  float *packed = NULL;
  if (rank == ROOT) {
    size_t full = N * N * sizeof(float);
    A = malloc(full);
    B = malloc(full);
    C = malloc(full);
    packed = malloc(full);
    matrix_fill(A, N);
    matrix_fill(B, N);
  }

  if (rank == ROOT) {
    packblk(A, packed, N, prows, pcols);
  }
  MPI_Scatter(packed, blk, MPI_FLOAT, Al, blk, MPI_FLOAT, ROOT, grid);

  if (rank == ROOT) {
    packblk(B, packed, N, prows, pcols);
  }
  MPI_Scatter(packed, blk, MPI_FLOAT, Bl, blk, MPI_FLOAT, ROOT, grid);

  /// ===============
  /// SUMMA
  /// ===============
  TimeSpend timer;

  MPI_Barrier(grid);
  start_time(&timer);
  for (int s = 0; s < steps; s++) {
    int k0 = s * panel;
    int aOwner = k0 / nb;
    int aOff = k0 % nb;

    if (mycol == aOwner) {
      for (int i = 0; i < mb; i++) {
        size_t panelIdx = (size_t)i * panel;
        size_t localIdx = (size_t)i * nb + aOff;
        memcpy(&Ap[panelIdx], &Al[localIdx], panel * sizeof(float));
      }
    }
    MPI_Bcast(Ap, mb * panel, MPI_FLOAT, aOwner, rowComm);

    int bOwner = k0 / mb;
    int bOff = k0 % mb;
    float *B = (myrow == bOwner) ? &Bl[(size_t)bOff * nb] : Bp;
    MPI_Bcast(B, nb * panel, MPI_FLOAT, bOwner, colComm);

    matmul(mb, nb, panel, Ap, panel, B, nb, Cl, nb, T);
  }
  MPI_Barrier(grid);
  finish_time(&timer);

  MPI_Gather(Cl, blk, MPI_FLOAT, packed, blk, MPI_FLOAT, 0, grid);
  if (rank == 0) {
    unpackblk(packed, C, N, prows, pcols);

    FILE *file = fopen(argv[3], "w");
    matrix_print(C, (size_t)N, file);
    fclose(file);

    fprintf(stdout, "%d,%dx%d,%ld\n", size, N, N, elapsed_time(&timer));
  }

  MPI_Comm_free(&rowComm);
  MPI_Comm_free(&colComm);
  MPI_Comm_free(&grid);
  MPI_Finalize();
  return 0;
}