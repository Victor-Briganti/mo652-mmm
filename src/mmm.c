#include "utils.h"
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//===------------------------------------------------------------------------===
// SUMMA (Scalable Universal Matrix Multiplication Algorithm)
//===------------------------------------------------------------------------===

/// Performs a retangular matrix multiplication using tiling.
///
/// m: Number of rows of Matrix A and C.
/// n: Number of column of Matrix B and C.
/// kd: Columns of A and rows of B.
/// A: Matrix of dimension m x kd.
/// lda: Leading Dimensions of matrix A.
/// B: Matrix of dimension kd x n.
/// ldb: Leading Dimensions of matrix B.
/// C: Accumulator matrix.
/// ldc: Leading Dimensions of matrix C.
/// T: Tile used for cache blocking.
static void gemm(int m, int n, int kd, const float *A, int lda, const float *B,
                 int ldb, float *restrict C, int ldc, int T) {
  for (int ii = 0; ii < m; ii += T)
    for (int jj = 0; jj < n; jj += T)
      for (int kk = 0; kk < kd; kk += T) {
        // Clamp tile boundaries for dimensions not divisible by T
        int ie = MIN(ii + T, m);
        int je = MIN(jj + T, n);
        int ke = MIN(kk + T, kd);
        for (int i = ii; i < ie; i++)
          for (int k = kk; k < ke; k++) {
            // Keep scalar in a register across the innermost loop
            float a = A[(size_t)i * lda + k];
            for (int j = jj; j < je; j++)
              C[(size_t)i * ldc + j] += a * B[(size_t)k * ldb + j];
          }
      }
}

/// Executes parallel matrix multiplication using the SUMMA algorithm.
///
/// Decompose the global matrix across a 2D cartesian grid of processes. In each
/// iteration, ranks broadcast a narrow slice ("panel") of A along their row and
/// a slice of B down their column, updating their local submatrix C.
///
/// Once all panels are processed, the distributed blocks of C are gathered and
/// unpacked into standard row-major order on the root rank.
static TimeSpend summa(MpiContext *ctx) {
  TimeSpend timer;

  // Syncrhonize all ranks before measuring time
  MPI_Barrier(ctx->cart.commGrid);
  start_time(&timer);
  for (int s = 0; s < ctx->data.steps; s++) {
    // Matrix coordinate where the tile begins
    int k0 = s * ctx->data.panel;

    // Defines the process number that owns this slice of A
    int aOwner = k0 / ctx->data.nb;
    // Local column offset inside the local block `Al`
    int aOffset = k0 % ctx->data.nb;

    // If this process is the owner, extract the vertical strip into Ap
    if (ctx->cart.y == aOwner) {
      // Here we are packing the column of the matrix in a contiguous
      // allocation.
      for (int i = 0; i < ctx->data.mb; i++) {
        size_t panelIdx = (size_t)i * ctx->data.panel;
        size_t localIdx = (size_t)i * ctx->data.nb + aOffset;
        memcpy(&ctx->data.Ap[panelIdx], &ctx->data.Al[localIdx],
               ctx->data.panel * sizeof(float));
      }
    }
    // Broadcast the Ap to all processes in the same row.
    MPI_Bcast(ctx->data.Ap, ctx->data.mb * ctx->data.panel, MPI_FLOAT, aOwner,
              ctx->cart.commRow);

    // Defines the process number that owns this slice of B
    int bOwner = k0 / ctx->data.mb;
    // Local column offset inside the local block Bl
    int bOffset = k0 % ctx->data.mb;

    // The owner already has contiguous rows in memory, so it points directly
    // into Bl. No-owners receive into the temporary Bl buffer
    float *B = (ctx->cart.x == bOwner)
                   ? &ctx->data.Bl[(size_t)bOffset * ctx->data.nb]
                   : ctx->data.Bp;
    MPI_Bcast(B, ctx->data.nb * ctx->data.panel, MPI_FLOAT, bOwner,
              ctx->cart.commCol);

    gemm(ctx->data.mb, ctx->data.nb, ctx->data.panel, ctx->data.Ap,
         ctx->data.panel, B, ctx->data.nb, ctx->data.Cl, ctx->data.nb,
         ctx->data.T);
  }

  // Collect local blocks back into the root process.
  // Each rank sends its local Cl into the root contiguous stage buffer
  MPI_Gather(ctx->data.Cl, ctx->data.blk, MPI_FLOAT, ctx->data.stageBuffer,
             ctx->data.blk, MPI_FLOAT, ROOT_RANK, ctx->cart.commGrid);
  if (ctx->isRoot) {
    unpack_block(ctx, ctx->data.C);
  }

  MPI_Barrier(ctx->cart.commGrid);
  finish_time(&timer);

  return timer;
}

//===------------------------------------------------------------------------===
// Main
//===------------------------------------------------------------------------===

int main(int argc, char **argv) {
  MpiContext ctx = mpi_init();

  int params[2] = {
      0, /* N */
      0  /* T */
  };
  if (ctx.isRoot) {
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
  MPI_Bcast(params, 2, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

  int N = params[0];
  int T = params[1];

  mpi_data_init(&ctx, N, T);

  if (ctx.isRoot) {
    pack_block(&ctx, ctx.data.A);
  }
  MPI_Scatter(ctx.data.stageBuffer, ctx.data.blk, MPI_FLOAT, ctx.data.Al,
              ctx.data.blk, MPI_FLOAT, ROOT_RANK, ctx.cart.commGrid);

  if (ctx.isRoot) {
    pack_block(&ctx, ctx.data.B);
  }
  MPI_Scatter(ctx.data.stageBuffer, ctx.data.blk, MPI_FLOAT, ctx.data.Bl,
              ctx.data.blk, MPI_FLOAT, ROOT_RANK, ctx.cart.commGrid);

  TimeSpend timer = summa(&ctx);

  if (ctx.isRoot) {
    FILE *file = fopen(argv[3], "w");
    matrix_print(ctx.data.C, (size_t)N, file);
    fclose(file);
    fprintf(stdout, "%d,%dx%d,%ld\n", ctx.worldSz, N, N, elapsed_time(&timer));
  }

  mpi_finish(&ctx);
  return 0;
}