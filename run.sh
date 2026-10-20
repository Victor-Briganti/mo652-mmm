#!/bin/bash
#PBS -N mmm
#PBS -q testes
#PBS -l nodes=1:ppn=4
#
#   #PBS -q paralela
#   #PBS -l nodes=8:ppn=128
#
#   #PBS -q parexp
#   #PBS -l nodes=11:ppn=48

BIN=$PBS_O_WORKDIR/build
RESULTS=${RESULTS:-$PBS_O_WORKDIR/results}
CSV=${CSV:-$RESULTS/results.csv}
LOG=${LOG:-$RESULTS/run.log}

SIZES=${SIZES:-"1024 4096 8192"}
RUNS=3
TILES="32 64 128"

cd $PBS_O_WORKDIR

source $WORKDIR/spack/share/spack/setup-env.sh
spack env activate trab02

# Number of total cores (MAXP) e of cores per node (CPN)
MAXP=$(wc -l < $PBS_NODEFILE)
CPN=$(sort $PBS_NODEFILE | uniq -c | head -1 | awk '{print $1}')

# Creates a list with the MAX number of process this system supports. The list increase in powers of 2.
PROCS=""
for ((np = 1; np <= MAXP; np *= 2)); do
  PROCS="$PROCS $np"
done

# Creates a list with the MAX number of threads this system supports. The list increase in powers of 2.
THREADS=""
for ((t = 1; t <= CPN; t *= 2)); do
  THREADS="$THREADS $t"
done

mkdir -p $RESULTS $(dirname $CSV) $(dirname $LOG)
if [ ! -f $CSV ]; then
  echo "run,version,procs,grid,threads,N,tile,time_ns,verify,max_abs,max_rel" > $CSV
fi

# Usage: bench <run> <cmd...>
bench() {
  local run=$1
  shift
  echo "[$(date +%T)] run=$run: $*" | tee -a $LOG
  "$@" 2>> $LOG | sed "s/^/$run,/" >> $CSV
}

# Usage: measure <cmd...>, RUNS executions where only the first one is verified
measure() {
  for ((r = 1; r <= RUNS; r++)); do
    bench $r "$@" $N -v
  done
}

# Usage: mpi <procs> <threads>, ranks per node leave room for their threads
mpi() {
  echo "mpirun -np $1 -ppn $((CPN / $2)) -machinefile $PBS_NODEFILE" \
    "-genv OMP_NUM_THREADS $2"
}

for N in $SIZES; do
  measure $BIN/mms

  for np in $PROCS; do
    cols=1
    while ((cols * cols * 4 <= np)); do
      cols=$((cols * 2))
    done

    for tile in $TILES; do
      if ((tile <= N / cols)); then
        measure $(mpi $np 1) $BIN/mmm -t $tile
      fi
    done

    for t in $THREADS; do
      if ((np * t <= MAXP)); then
        echo "threads $2"
        measure $(mpi $np $t) $BIN/mmmb
      fi
    done
  done
done
