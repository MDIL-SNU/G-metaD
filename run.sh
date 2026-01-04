#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=rapids          # Partition name (skylake)
##
#SBATCH --job-name="test"
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)


PORT=8021
TOTAL_CORES=32
STEPS=1000
RESTART_FREQ=100
POSCAR_TEMPLATE="POSCAR_template"
CALC_MODE="vasp"
VASP_CMD="mpirun -np 28 /TGM/Apps/VASP/VASP_BIN/6.4.2/vasp.6.4.2.avx512.std.x"
ML_MODEL_PATH="trained_model.pt"
LMP_BIN="lmp"


cleanup() {

    sleep 5

    if [[ -n "DRIVER_PID" ]]; then
        kill -TERM $DRIVER_PID 2>/dev/null
    fi
    if [[ -n "LAMMPS_PID" ]]; then
        kill -TERM $LAMMPS_PID 2>/dev/null
    fi
}

trap cleanup EXIT INT TERM
 
python -u g_metad_driver.py \
    --mdi "-role DRIVER -name driver -method TCP -port ${PORT}" \
    --poscar_template "${POSCAR_TEMPLATE}" \
    --calculator ${CALC_MODE} \
    --vasp_cmd "${VASP_CMD}" \
    --ml_model "${ML_MODEL_PATH}" \
    --steps ${STEPS} \
    --restart_freq ${RESTART_FREQ} \
    --max_try 4 \
    1> driver.log \
    2> driver.err &

DRIVER_PID=$!

sleep 5

mpirun -np 4 lmp \
    -mdi "-role ENGINE -name LAMMPS -method TCP -port 8021 -hostname localhost" \
    -in in.client \
    > lammps.log 2>&1 &

LAMMPS_PID=$!

wait -n $DRIVER_PID $LAMMPS_PID
EXIT_CODE=$?

exit $EXIT_CODE
