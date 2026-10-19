#!/bin/bash
#PBS -N mmm
#PBS -q testes
#PBS -l nodes=1:ppn=4
#PBS -o mmm.out
#PBS -e mmm.err
#
# Other queues (the process counts adapt to the allocation):
#   #PBS -q paralela
#   #PBS -l nodes=8:ppn=128
#
#   #PBS -q parexp
#   #PBS -l nodes=11:ppn=48

module purge
module load openblas/0.3.30-gcc-15.2.0 gcc/15.2.0 cmake/4.3.1 ninja/1.11.1-gcc-12.2.0

source $WORKDIR/spack/share/spack/setup-env.sh
spack env activate trab02


mpirun -genv OMP_NUM_THREADS=1 -np 1 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=2 -np 1 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=4 -np 1 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=1 -np 2 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=2 -np 2 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=4 -np 2 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=1 -np 4 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=2 -np 4 $WORKDIR/temp/mmm
mpirun -genv OMP_NUM_THREADS=4 -np 4 $WORKDIR/temp/mmm
