# MPI Matrix Multiplication (MMM)

This repository was created as part of the High Performance Computing course at Unicamp to better understand MPI and evaluate how workload division affects system performance.

## Activating Environment

The installation of the environment uisng spack, can be made with the following command:

```bash
spack env create trab02 .spack
spack env activate trab02
spack install
```

## Compilation

The command to compile the project is based on the following:

```bash
module purge
module load ninja/1.11.1-gcc-12.2.0 cmake/3.31.8-gcc-12.2.0 gcc/15.2.0 openblas/0.3.30-gcc-15.2.0
spack env activate trab02
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Executing

This project generates three binaries:

- `mms`: Serial execution (without tilling);
- `mmm`: MPI execution (with tilling);
- `mmmb`: MPI execution (with BLAS);

They can be executed as the following:

```bash
./build/mms 2000
mpirun -np 4 ./build/mmm 2000 -t 64 -v 
mpirun -np 8 ./build/mmmb 2000 -t 32 -v 
```

Where `2000` is the dimension of the matrix, the `-t` is the flag that specify the tile size (default 64). One last flag that is not showing in there is the flag `-o` that is used to store the matrix as a CSV file.

**Obs:** In the MPI executions the size of the matrix must be divisible by the the number of processes.
