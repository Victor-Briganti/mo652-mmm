#ifndef MM_SUMMA_H
#define MM_SUMMA_H

#include "utils.h"

//===------------------------------------------------------------------------===
// Variants
//===------------------------------------------------------------------------===

/// Local block multiplication: C (m x n) += A (m x kd) * B (kd x n).
///
/// lda, ldb, ldc: Leading dimensions of A, B and C.
/// T: Tile used for cache blocking (ignored by kernels that do their own).
typedef void (*LocalGemm)(int m, int n, int kd, const float *A, int lda,
                          const float *B, int ldb, float *restrict C, int ldc,
                          int T);

typedef struct SummaVariant {
  const char *name; // Version of the program being executed
  LocalGemm gemm;   // Kernel used for the local block products
  int threads;      // Number of threads used by the kernel.
  bool usesTile;
} SummaVariant;

//===------------------------------------------------------------------------===
// SUMMA (Scalable Universal Matrix Multiplication Algorithm)
//===------------------------------------------------------------------------===

/// Scatters A and B from the root into the local blocks of every rank.
static inline void summa_distribute(MpiContext *ctx) {
  if (ctx->isRoot) {
    pack_block(ctx, ctx->data.A);
  }
  MPI_Scatter(ctx->data.stageBuffer, ctx->data.blk, MPI_FLOAT, ctx->data.Al,
              ctx->data.blk, MPI_FLOAT, ROOT_RANK, ctx->cart.commGrid);

  if (ctx->isRoot) {
    pack_block(ctx, ctx->data.B);
  }
  MPI_Scatter(ctx->data.stageBuffer, ctx->data.blk, MPI_FLOAT, ctx->data.Bl,
              ctx->data.blk, MPI_FLOAT, ROOT_RANK, ctx->cart.commGrid);
}

/// Executes parallel matrix multiplication using the SUMMA algorithm.
///
/// Decompose the global matrix across a 2D cartesian grid of processes. In each
/// iteration, ranks broadcast a narrow slice ("panel") of A along their row and
/// a slice of B down their column, updating their local submatrix C with
/// `gemm`.
///
/// Once all panels are processed, the distributed blocks of C are gathered and
/// unpacked into standard row-major order on the root rank. The measured time
/// covers the panel loop, the gather and the unpack, but not the initial
/// scatter.
static inline TimeSpend summa_exec(MpiContext *ctx, LocalGemm gemm) {
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
// Driver
//===------------------------------------------------------------------------===

static inline int summa(int argc, char **argv, const SummaVariant *variant) {
  MpiContext ctx = mpi_init();

  Parameters opts = {0};
  int params[3] = {0};
  if (ctx.isRoot) {
    if (!parse_options(argc, argv, &opts)) {
      usage(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      return -1;
    }

    params[0] = opts.N;
    params[1] = opts.T;
    params[2] = opts.verify;
  }
  MPI_Bcast(params, 3, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

  opts.N = params[0];
  opts.T = params[1];
  opts.verify = params[2];

  mpi_data_init(&ctx, opts.N, opts.T);

  int maxTile = MAX(ctx.data.mb, ctx.data.nb);
  if (variant->usesTile && opts.T > maxTile) {
    if (ctx.isRoot) {
      fprintf(stderr,
              "Tile %d is larger than the local block (%dx%d) of each rank\n",
              opts.T, ctx.data.mb, ctx.data.nb);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  summa_distribute(&ctx);

  TimeSpend timer = summa_exec(&ctx, variant->gemm);

  int status = EXIT_SUCCESS;
  if (ctx.isRoot) {
    Verification v = {0};
    if (opts.verify) {
      openblas_set_num_threads(1);
      v = matrix_verify(ctx.data.C, ctx.data.A, ctx.data.B, (size_t)opts.N);
      status = v.ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (opts.output) {
      FILE *file = fopen(opts.output, "w");
      if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", opts.output);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return -1;
      }

      matrix_print(ctx.data.C, opts.N, file);
    }

    print_result(variant->name, ctx.worldSz, ctx.cart.procRows,
                 ctx.cart.procCols, variant->threads, opts.N,
                 variant->usesTile ? opts.T : 0, elapsed_time(&timer),
                 opts.verify ? &v : NULL);
  }

  mpi_finish(&ctx);
  return status;
}

#endif // MM_SUMMA_H
