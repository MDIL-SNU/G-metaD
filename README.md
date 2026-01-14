# 1. What is G-metaD?
G-metaD (Metadynamics sampling in atomic environment space) is an enhanced sampling method
designed to efficiently generate training datasets for machine learning interatomic potentials (MLIP).

Unlike standard molecular dynamics, which often gets trapped in local minima, 
G-metaD uses atom-centered symmetry function vectors (G-vectors) as collective variables (CVs) for metadynamics.
By adding a history-dependent bias potential to these G-vectors, the system is encouraged to explore rare events
and high-energy configurations that are crucial for training robust ML potentials.

Reference: D. Yoo, J. Jung, W. Jeong, and S. Han, Metadynamics sampling in atomic environment space for collecting training data for machine learning potentials, *npj Computational Materials* **7**, 131 (2021), [[https://doi.org/10.1038/s41524-021-00595-5](https://doi.org/10.1038/s41524-021-00595-5)]

# 2. Overview
This package interfaces LAMMPS (Driver) and VASP/MLFF (Engine) using MDI (MolSSI Driver Interface) protocol.

* LAMMPS: Acts as the MD driver. It integrates the equations of motion and applies the G-metaD bias potential.
* Python Engine: Acts as the energy/force calculator. It wraps VASP(DFT) or MLFF, handles unit conversions, and cimmunicates with LAMMPS via TCP sockets.

# 3. Prerequisites
* LAMMPS (tested at 22Jul2025)
* Eigen Library (tested at 3.4.1)
* Python 3.8+
* (Optional) VASP (for DFT calculations)
* (Optional) MLFF supports ASE calculator

# 4. Installation & Build

## 4.1 Install Eigen library
G-metaD uses the `Eigen` library for matrix operations. You do not need to compile `Eigen`,
but you must download the source code from the [Eigen official website](https://libeigen.gitlab.io/).

## 4.2 Add G-metaD source codes
```bash
cp pair_mtd.* symmetry_function.h /path/to/lammps/src
```

## 4.3 Compile LAMMPS with CMake
Create a build directory and run CMake. You must enable the MDI package, MPI, and specify the Eigen include directory
```bash
cd /path/to/lammps
mkdir build && cd build

cmake ../cmake \
    -D DOWNLOAD_MDI=yes \
    -D PKG_MDI=yes \
    -D BUILD_MPI=yes \
    -D CMAKE_CXX_FLAGS="-I/path/to/eigen"

cmake --build . --target install
```
> [!NOTE]
> If it fails due to missing Eigen headers, ensure the `-I/path/to/eigen` path correctly points to the directory containing the `Eigen` folder.

## 4.4 Python environment setup
```bash
pip install pymdi
```
Download the MLFF you want to use.

# 5. Usage & CPU resource allocation
> [!IMPORTANT]
> Since LAMMPS and VASP run concurrently, you must split your available CPU cores between them.
> They should not share the same cores (oveersubscription), as this causes severe performance degradation.
> * LAMMPS (Driver): Extremely lightweight. It only updates positions and calculates the bias potential.
> * VASP (Engine): Computational bottleneck. It requires most CPU resources.

## 5.1 Prepare input files
Ensure the following files are present:
* in.client: LAMMPS input script
* POSCAR_template: Structure template
* g_metad_engine.py: Python MDI Engine
* run.sh Execution script
* (VASP): INCAR, POTCAR, KPOINTS
* (MLFF): model.pt

## 5.2 Configure run.sh
Modify the `-np` (number of processes flags i`run.sh` to adhere to the splitting rule.
```bash
export OMP_NUM_THREADS=1

PORT=8021
POSCAR_TEMPLATE="POSCAR_template"
CALC_MODE="vasp"
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
```

# 6. Input script example

## 6.1 G-metaD pair style
```text
# Syntax: pair_style mtd [cond_num] [N_elements] [Symbol] [Gaussian height] [Gaussian width] [Update interval] [N_types] [Bias flag]
pair_style mtd 1e-4 1 &
                Si 0.001 1.0 20 &
                1 &
                1.0
```
If the system consist of two elements such as GeTe, `in.client` is following:
```text
pair_style mtd 1e-4 2 &
                Ge 0.001 1.0 20 &
                Te 0.001 1.0 20 &
                2 &
                1.0 1.0
```

## 6.2 MDI fix configuration
Use `fix mdi/qm` to retrieve energy, forces, and stress.
``` text
fix my_mdi all mdi/qm virial yes
```

# 7. Advanced tips: controlling bias accumulation
The `pair_modify tail` command controls whether the bias potential is updated (accumulated) or kept static.

## 7.1 Dynamic bias (accumulation on)
To continuously add new Gaussian hills and flatten the free energy surface (FES), use the default setting (`tail no`).
* Behavior: The bias potential grows over time.
* Use case: When you want to push the system out of local minima of fill the basins.
```text
pair_modify tail no
```

## 7.2 Static bias (accumulation off)
To stop adding new hills and sample configurations on the currently accumulated bias potential, use `tail yes`.
* Behavior: The bias potential remains fixed (frozen).
* Use case: When the FES is sufficiently flattened, and you want to sample equilibrium distributions on the modified surface.
```text
pair_modify tail yes
```

# 8. Troubleshooting

## Q1. `Could not bind socket` error
* Cause: A previous job left a zombie process holding the port.
* Solution: Kill all processes or change the port in `run.sh`.

## Q2. VASP fails (symmetry error)
* Cause: During MD, structural distortions may cause VASP to fail in determining symmetry, leading to crashes.
* Solution: Increase the `SYMPREC` parameter in your `INCAR` file for symmetry tolerance.
