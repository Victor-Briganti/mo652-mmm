#ifndef MM_UTILS_H
#define MM_UTILS_H

#include <cblas.h>
#include <float.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ROOT_RANK (0)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/// Columns of the CSV line printed to stdout by every version.
#define RESULT_HEADER                                                          \
  "version,procs,grid,threads,N,tile,time_ns,verify,max_abs,max_rel"

//===------------------------------------------------------------------------===
// Structs
//===------------------------------------------------------------------------===

typedef struct TimeSpend {
  struct timespec before;
  struct timespec after;
} TimeSpend;

typedef struct Parameters {
  int N;              // Global matrix dimension (N x N)
  int T;              // Cache tile size of the non-BLAS kernels
  bool verify;        // Check the result against the serial reference
  const char *output; // Optional CSV dump of C
} Parameters;

typedef struct Verification {
  bool ok;
  double maxAbs; // max |C - Ref|
  double maxRel; // maxAbs / max |Ref|
  double tol;    // Largest accepted maxRel
} Verification;

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

static inline void usage(const char *prog) {
  fprintf(
      stderr,
      "Usage: %s [-t tile] [-v] [-o out.csv] N\n"
      "  -t tile     cache tile of the local kernel (default 64)\n"
      "  -v          verify against the serial reference\n"
      "  -o file     write the resulting matrix C as CSV\n"
      "N must be divisible by both dimensions of the process grid.\n"
      "The number of threads used by OpenBLAS are configured with the env var "
      "OMP_NUM_THREADS.\n",
      prog);
}

/// Parses the command line shared by all versions. Returns false on error.
static inline bool parse_options(int argc, char **argv, Parameters *opts) {
  *opts = (Parameters){.T = 64};

  int c;
  while ((c = getopt(argc, argv, "t:vo:h")) != -1) {
    switch (c) {
    case 't':
      opts->T = atoi(optarg);
      break;
    case 'v':
      opts->verify = true;
      break;
    case 'o':
      opts->output = optarg;
      break;
    default:
      return false;
    }
  }

  if (optind != argc - 1) {
    return false;
  }
  opts->N = atoi(argv[optind]);

  if (opts->N <= 0 || opts->T <= 0) {
    fprintf(stderr, "N and tile must be positive\n");
    return false;
  }

  return true;
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

/// Serial matrix multiplication.
///
/// Uses the i-k-j order so the innermost loop streams rows of B and C while
/// A[i][k] stays in a register.
static inline void sgemm(const float *A, const float *B, float *restrict C,
                         size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t k = 0; k < N; k++) {
      float a = A[i * N + k];
      for (size_t j = 0; j < N; j++) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}

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
inline static void gemm(int m, int n, int kd, const float *A, int lda,
                        const float *B, int ldb, float *restrict C, int ldc,
                        int T) {
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

static void cgemm(int m, int n, int kd, const float *A, int lda, const float *B,
                  int ldb, float *restrict C, int ldc, int T) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, kd, // M, N, K
              1.0f,                                                // alpha
              A, lda, // A and its leading dimension
              B, ldb, // B and its leading dimension
              1.0f,   // beta = 1 -> accumulate into C
              C, ldc);
}

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
      // Generates a pseudo random float between [-4, 5)
      float whole = (float)((int)((i * 7 + j * 13) % 9) - 4);
      float frac = (float)((i * 7919 + j * 104729) % 10000) / 10000.0f;
      M[i * N + j] = whole + frac;
    }
  }
}

/// Compares the result with the serial reference
static inline Verification matrix_verify(const float *C, const float *A,
                                         const float *B, size_t N) {
  Verification v = {0};

  float *R = calloc(N * N, sizeof(float));
  sgemm(A, B, R, N);

  double refMax = 0.0;
  for (size_t idx = 0; idx < N * N; idx++) {
    double diff = (double)C[idx] - (double)R[idx];
    double ref = (double)R[idx];

    // Stores the max absolute difference found during comparison
    v.maxAbs = MAX(v.maxAbs, diff < 0 ? -diff : diff);

    // Stores the max absolute value in the entire reference matrix
    refMax = MAX(refMax, ref < 0 ? -ref : ref);
  }
  free(R);

  // Calculates the max relative difference
  v.maxRel = refMax > 0.0 ? v.maxAbs / refMax : v.maxAbs;
  // Calculates the maximum theoretical accumulation of round-off errors during
  // matrix multiplication.
  v.tol = (double)N * FLT_EPSILON;
  v.ok = v.maxRel <= v.tol;

  fprintf(stderr, "Verification %s: max_abs=%.3e max_rel=%.3e tol=%.3e\n",
          v.ok ? "OK" : "FAILED", v.maxAbs, v.maxRel, v.tol);
  return v;
}

static inline void print_result(const char *version, int procs, int rows,
                                int cols, int threads, int N, int tile,
                                int64_t ns, const Verification *v) {
  printf("%s,%d,%dx%d,%d,%d,%d,%ld", version, procs, rows, cols, threads, N,
         tile, ns);
  if (v != NULL) {
    printf(",%d,%.3e,%.3e\n", v->ok, v->maxAbs, v->maxRel);
  } else {
    printf(",NA,NA,NA\n");
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
  int coords[2] = {0, 0};
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

/// Splits the N x N matrices over the process grid.
static inline void mpi_data_init(MpiContext *ctx, int N, int T) {
  if (N % ctx->cart.procRows != 0 || N % ctx->cart.procCols != 0) {
    if (ctx->isRoot) {
      fprintf(
          stderr,
          "Matrix of size %d is not divisble by the grid dimensions (%dx%d)\n",
          N, ctx->cart.procRows, ctx->cart.procCols);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

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
    size_t full = (size_t)N * N * sizeof(float);
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

/// Packs an NxN matrix from row major into a contiguous layout, ready for
/// MPI_Scatter.
static inline void pack_block(MpiContext *ctx, const float *original) {
  size_t rowBytes = (size_t)ctx->data.nb * sizeof(float);

  for (int row = 0; row < ctx->cart.procCols * ctx->cart.procRows; row++) {
    // Map linear rank indes into 2D cartesian coordinates (gridRow, gridCol)
    int gridRow = row / ctx->cart.procCols;
    int gridCol = row % ctx->cart.procCols;

    // Calculate top-left element coordinates of this block in the global
    // matrix.
    size_t firstRow = (size_t)gridRow * ctx->data.mb;
    size_t firstCol = (size_t)gridCol * ctx->data.nb;

    // Start of the contiguous buffer for this process block
    float *block = &ctx->data.stageBuffer[(size_t)row * ctx->data.blk];

    // Copy the block row-by-row into contiguous memory
    for (int i = 0; i < ctx->data.mb; i++) {
      const float *srcRow = &original[(firstRow + i) * ctx->data.N + firstCol];
      float *dstRow = &block[(size_t)i * ctx->data.nb];
      memcpy(dstRow, srcRow, rowBytes);
    }
  }
}

/// Unpacks contiguous process blocks from stageBuffer back into NxN matrix in
/// standard row-major layout.
static inline void unpack_block(const MpiContext *ctx, float *original) {
  size_t rowBytes = (size_t)ctx->data.nb * sizeof(float);

  // Iterate over every process block in the Cartesian grid
  for (int row = 0; row < ctx->cart.procCols * ctx->cart.procRows; row++) {
    // Map linear rank index to 2D cartesian coordinates (gridRow, gridCol)
    int gridRow = row / ctx->cart.procCols;
    int gridCol = row % ctx->cart.procCols;

    // Calculate top-left element coordinates of this block in the global
    // matrix.
    size_t firstRow = (size_t)gridRow * ctx->data.mb;
    size_t firstCol = (size_t)gridCol * ctx->data.nb;

    // Start of this process contiguous block in stageBuffer
    const float *block = &ctx->data.stageBuffer[(size_t)row * ctx->data.blk];

    // Copy the block row-by-row back into the global row-major matrix
    for (int i = 0; i < ctx->data.mb; i++) {
      const float *srcRow = &block[(size_t)i * ctx->data.nb];
      float *dstRow = &original[(firstRow + i) * ctx->data.N + firstCol];
      memcpy(dstRow, srcRow, rowBytes);
    }
  }
}

#endif // MM_UTILS_H
