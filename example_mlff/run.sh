export OMP_NUM_THREADS=1

PORT=8021
POSCAR_TEMPLATE="POSCAR_template"
CALC_MODE="mlff"
VASP_CMD="mpirun -np 28 vasp_std"
ML_MODEL_PATH="trained_model.pt"
LMP_BIN="lmp"


echo "Starting LAMMPS Driver..."
mpirun -np 4 lmp \
    -mdi "-role DRIVER -name DRIVER -method TCP -port ${PORT}" \
    -in in.client \
    > lammps.log 2>&1 &

DRIVER_PID=$!
echo "DRIVER PID: $DRIVER_PID"

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

trap "kill -9 $DRIVER_PID $ENGINE_PID 2>/dev/nulll; exit" SIGINT SIGTERM

while true; do
    if ! kill -0 $DRIVER_PID 2> /dev/null; then
        kill -9 $ENGINE_PID 2> /dev/null
        break
    fi

    if ! kill -0 $ENGINE_PID 2> /dev/null; then
        kill -9 $DRIVER_PID 2> /dev/null
        break
    fi

    sleep 1
done
