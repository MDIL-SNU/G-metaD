#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=rapids          # Partition name (skylake)
##
#SBATCH --job-name="test"
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)


export OMP_NUM_THREADS=1

PORT=8021
TOTAL_CORES=32
POSCAR_TEMPLATE="POSCAR_template"
CALC_MODE="vasp"
VASP_CMD="mpirun -np 28 /TGM/Apps/VASP/VASP_BIN/6.4.2/vasp.6.4.2.avx512.std.x"
ML_MODEL_PATH="trained_model.pt"
LMP_BIN="lmp"


cleanup() {
    if [[ -n "LAMMPS_PID" ]]; then kill -TERM $LAMMPS_PID 2>/dev/null; fi
    if [[ -n "ENGINE_PID" ]]; then kill -TERM $ENGINE_PID 2>/dev/null; fi
}
trap cleanup EXIT INT TERM
 
echo "Starting LAMMPS (Driver)..."
mpirun -np 4 lmp \
    -mdi "-role DRIVER -name LAMMPS -method TCP -port ${PORT}" \
    -in in.client \
    > lammps.log 2>&1 &

LAMMPS_PID=$!
echo "LAMMPS PID: $LAMMPS_PID"

sleep 5

echo "Starting Python Engine..."
python -u g_metad_engine.py \
    --mdi "-role ENGINE -name ${CALC_MODE} -method TCP -port ${PORT} -hostname localhost"\
    --poscar_template "${POSCAR_TEMPLATE}" \
    --calculator ${CALC_MODE} \
    --vasp_cmd "${VASP_CMD}" \
    --ml_model "${ML_MODEL_PATH}" \
    --max_try 4 \
    1> engine.log 2> engine.err &

ENGINE_PID=$!
echo "Python Engine PID: $ENGINE_PID"

wait -n $LAMMPS_PID $ENGINE_PID
EXIT_CODE=$?

exit $EXIT_CODE
