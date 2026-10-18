#ifndef MM_UTILS_H
#define MM_UTILS_H

#include <mpi.h>
#include <mpi_proto.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROOT_RANK (0)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

//===------------------------------------------------------------------------===
// Structs
//===------------------------------------------------------------------------===

typedef struct TimeSpend {
  struct timespec before;
  struct timespec after;
} TimeSpend;

typedef struct MpiCart2D {
  // 2D process grid dimensions
  int procRows;
  int procCols;

  // Coordinates of this node on the grid
  int x, y;

  // Cartesian communicators
  MPI_Comm commGrid; // 2D Cartesian topology communicator
  MPI_Comm commRow;  // Sub-communicator for row broadcasts
  MPI_Comm commCol;  // Sub-communicator for columns broadcasts
} MpiCart2D;

typedef struct MpiMatrix {
  int N; // Global matrix dimension (N x N)
  int T; // Cache tile size

  // Local process block dimensions
  int mb;  // Number of rows per local block (N / procRows)
  int nb;  // Number of columns per local block (N / procCols)
  int blk; // Total elements in local block (mb * nb)

  // SUMMA algorithmic parameters
  int panel; // Width dimension of the submatrix used in the multiplication
  int steps; // Total number of iterations based on the panel size

  // Persistent local sub-matrices
  float *Al;
  float *Bl;
  float *Cl;

  // Working panel buffers
  float *Ap;
  float *Bp;

  // Full matrices (used by the root)
  float *A;
  float *B;
  float *C;
  float *stageBuffer; // Buffer used to order the matrices for communication
} MpiData;

typedef struct MpiContext {
  int rank;
  int worldSz;
  bool isRoot;
  MpiCart2D cart;
  MpiData data;
} MpiContext;

//===------------------------------------------------------------------------===
// Common Utils
//===------------------------------------------------------------------------===

static inline int gcd(int a, int b) {
  while (b != 0) {
    int tmp = a % b;
    a = b;
    b = tmp;
  }

  return a;
}

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
// MPI Utils
//===------------------------------------------------------------------------===

static inline MpiContext mpi_init() {
  MpiContext ctx = {0};

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.worldSz);

  ctx.isRoot = (ctx.rank == ROOT_RANK);

  // Initialize cartesian communication
  int dims[2] = {0, 0};
  MPI_Dims_create(ctx.worldSz, 2, dims);
  ctx.cart.procRows = dims[0];
  ctx.cart.procCols = dims[1];

  // Period defines if the communcation wraps around.
  int periods[2] = {0, 0}; // (in this case it does not)
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &ctx.cart.commGrid);

  // Gets the coords of this node in the new system.
  int coords[2] = {};
  MPI_Cart_coords(ctx.cart.commGrid, ctx.rank, 2, coords);
  ctx.cart.x = coords[0];
  ctx.cart.y = coords[1];

  // Defines teh coomunication channel for the row and the column
  int dimsRow[2] = {0, 1}; // Groups all nodes in the same row in a channel
  int dimsCol[2] = {1, 0}; // Groups all nodes in the same col in a channel
  MPI_Cart_sub(ctx.cart.commGrid, dimsRow, &ctx.cart.commRow);
  MPI_Cart_sub(ctx.cart.commGrid, dimsCol, &ctx.cart.commCol);

  return ctx;
}

static inline void mpi_finish(MpiContext *ctx) {
  free(ctx->data.Al);
  free(ctx->data.Bl);
  free(ctx->data.Cl);
  free(ctx->data.Ap);
  free(ctx->data.Bp);
  free(ctx->data.A);
  free(ctx->data.B);
  free(ctx->data.C);
  free(ctx->data.stageBuffer);
  MPI_Comm_free(&ctx->cart.commRow);
  MPI_Comm_free(&ctx->cart.commCol);
  MPI_Comm_free(&ctx->cart.commGrid);
  MPI_Finalize();
}

static inline void mpi_data_init(MpiContext *ctx, int N, int T) {
  // Block divisions
  ctx->data.N = N;
  ctx->data.T = T;
  ctx->data.mb = N / ctx->cart.procRows;
  ctx->data.nb = N / ctx->cart.procCols;
  ctx->data.blk = ctx->data.mb * ctx->data.nb;
  ctx->data.panel = gcd(ctx->data.mb, ctx->data.nb);
  ctx->data.steps = N / ctx->data.panel;

  // Matrices allocation
  ctx->data.Al = malloc(sizeof(float) * ctx->data.blk);
  ctx->data.Bl = malloc(sizeof(float) * ctx->data.blk);
  ctx->data.Cl = calloc(ctx->data.blk, sizeof(float));
  ctx->data.Ap = malloc(sizeof(float) * ctx->data.mb * ctx->data.panel);
  ctx->data.Bp = malloc(sizeof(float) * ctx->data.nb * ctx->data.panel);

  if (ctx->isRoot) {
    size_t full = N * N * sizeof(float);
    ctx->data.A = malloc(full);
    ctx->data.B = malloc(full);
    ctx->data.C = malloc(full);
    ctx->data.stageBuffer = malloc(full);

    matrix_fill(ctx->data.A, N);
    matrix_fill(ctx->data.B, N);
  } else {
    ctx->data.A = NULL;
    ctx->data.B = NULL;
    ctx->data.C = NULL;
    ctx->data.stageBuffer = NULL;
  }
}

static inline void pack_block(MpiContext *ctx, const float *original) {
  size_t rowBytes = (size_t)ctx->data.nb * sizeof(float);

  for (int row = 0; row < ctx->cart.procCols * ctx->cart.procRows; row++) {
    int gridRow = row / ctx->cart.procCols;
    int gridCol = row % ctx->cart.procCols;

    size_t firstRow = (size_t)gridRow * ctx->data.mb;
    size_t firstCol = (size_t)gridCol * ctx->data.nb;

    float *block = &ctx->data.stageBuffer[row * ctx->data.blk];

    for (int i = 0; i < ctx->data.mb; i++) {
      const float *srcRow = &original[(firstRow + i) * ctx->data.N + firstCol];
      float *dstRow = &block[(size_t)i * ctx->data.nb];
      memcpy(dstRow, srcRow, rowBytes);
    }
  }
}

static inline void unpack_block(const MpiContext *ctx, float *original) {
  size_t rowBytes = (size_t)ctx->data.nb * sizeof(float);

  for (int row = 0; row < ctx->cart.procCols * ctx->cart.procRows; row++) {
    int gridRow = row / ctx->cart.procCols;
    int gridCol = row % ctx->cart.procCols;

    size_t firstRow = (size_t)gridRow * ctx->data.mb;
    size_t firstCol = (size_t)gridCol * ctx->data.nb;

    const float *block = &ctx->data.stageBuffer[row * ctx->data.blk];

    for (int i = 0; i < ctx->data.mb; i++) {
      const float *srcRow = &block[(size_t)i * ctx->data.nb];
      float *dstRow = &original[(firstRow + i) * ctx->data.N + firstCol];
      memcpy(dstRow, srcRow, rowBytes);
    }
  }
}

#endif // MM_UTILS_H