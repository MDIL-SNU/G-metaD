import sys
import os
import shutil
import subprocess
import time
import argparse
import re
import xml.etree.ElementTree as ET
from glob import glob
import mdi


class VASPCalculator:
    def __init__(self, vasp_cmd, poscar_template="POSCAR_template", max_try=4):
        self.vasp_cmd = vasp_cmd
        self.poscar_template = poscar_template
        self.max_try = max_try
        self.natoms, self.ntypes, _, self.natoms_per_type = self.vasp_setup()

    def _vasp_setup(self):
        if not os.path.exists(self.poscar_template):
            print(f"Error: {self.poscar_template} not found.")
            sys.exit(1)
        with open(self.poscar_template, "r") as f:
            lines = f.readlines()
        natoms_per_type = [int(x) for x in lines[6].split()]
        ntypes = len(natoms_per_type)
        natoms = sum(natoms_per_type)
        box = [0.0]*9
        return natoms, ntypes, box, natoms_per_type

    def write_poscar(self, coords, box, types):
        """Write POSCAR based on current coordinates from LAMMPS"""
        with open(self.poscar_template, "r") as f:
            template = f.readlines()

        with open("POSCAR", "w") as f:
            # Header from template
            f.write(template[0])
            f.write(template[1])

            # Box is usually [lx, ly, lz, xy, xz, yz, origin_x, origin_y, origin_z]
            f.write(f" {box[0]:12.8f} {box[1]:12.8f} {box[2]:12.8f}\n")
            f.write(f" {box[3]:12.8f} {box[4]:12.8f} {box[5]:12.8f}\n")
            f.write(f" {box[6]:12.8f} {box[7]:12.8f} {box[8]:12.8f}\n")

            f.write(template[5])
            f.write(template[6])
            f.write("Cartesian\n")

            # Coords are usually flat list [x1, y1, z1, x2, y2, z2...]
            atom_data = []
            for i in range(self.natoms):
                atom_data.append({
                    'type': types[i],
                    'x': coords[3*i],
                    'y': coords[3*i+1],
                    'z': coords[3*i+2]
                })

            atom_data.sort(key=lambda x: x['type'])

            for atom in atom_data:
                f.write(f" {atom['x']:12.8f} {atom['y']:12.8f} {atom['z']:12.8f}\n")

    def read_vasprun(self):
        """Parse vasprun.xml for energy, forces, virial"""
        try:
            tree = ET.parse("vasprun.xml")
            root = tree.getroot()
            calcs = root.findall("calculation")
            if not calcs:
                return 0, [], [], False

            last_calc = calcs[-1]
            scsteps = last_calc.findall("scstep")
            converged = True
            if len(scsteps) >= 60:
                converged = False

            # Energy
            e_child  last_calc.find("energy/i[@name='e_fr_energy']")
            if e_child is None:
                return 0, [], [], False
            e_free = float(e_child.text)

            # Forces & stress
            forces = []
            virial = []
            stress = []
            varrays = last_calc.findall("varray")
            for v in varrays:
                if v.attrib["name"] == "forces":
                    for line in v.findall("v"):
                        forces.extend([float(x) for x in line.text.split()])
                if v.attrib["name"] == "stress":
                    for line in v.findall("v"):
                        stress.append([float(x) for x in line.text.split()])
            virial[0] = stress[0][0]
            virial[1] = stress[1][1]
            virial[2] = stress[2][2]
            virial[3] = 0.5 * (stress[0][1] + stress[1][0])
            virial[4] = 0.5 * (stress[0][2] + stress[2][0])
            virial[5] = 0.5 * (stress[1][2] + stress[2][1])

            return e_free, forces, virial, converged

        except Exception as e:
            print(f"Error reading vasprun.xml: {e}")
            return 0, [], [], False

    def run_calculation(self, coords, box, types):
        self.write_poscar(coords, box, types)

        if not os.path.exists("INCAR_step_start"):
            shutil.copy2("INCAR", "INCAR_step_start")

        ntry = 0
        while True:
            ntry += 1
            try:
                subprocess.check_output(self.vasp_cmd, stderr=subprocess.STDOUT, shell=True)
                energy, forces_sorted, virial, converged = self.read_vasprun()

                if converged:
                    self._backup_outcar()
                    shutil.copy2("INCAR_step_start", "INCAR")
                    break
                else:
                    print(f"VASP Warning: Convergence failed (Attempt {ntry}/{self.max_try})")
                    if ntry >= self.max_try:
                        print("VASP Error: All convergence strategies failed.")
                        sys.exit(1)

                    self._apply_convergence_strategy(ntry + 1)

            except subprocess.CalledProcessError:
                if ntry >= self.max_try:
                    sys.exit(1)
                time.sleep(10)

            atom_indices = list(range(self.natoms))
            sorted_indices = sorted(atom_indices, key=lambda k: types[k])
            forces_ordered = [0.0] * (self.natoms * 3)
            for i, original_idx in enumerate(sorted_indices):
                forces_ordered[3 * original_idx + 0] = forces_sorted_by_type[3 * i + 0]
                forces_ordered[3 * original_idx + 1] = forces_sorted_by_type[3 * i + 1]
                forces_ordered[3 * original_idx + 2] = forces_sorted_by_type[3 * i + 2]

            return energy, forces_ordered, virial

    def _backup_outcar(self):
        outcars = glob("data/OUTCAR_*")
        idx = len(outcars)
        if not os.path.isdir("data"):
            os.mkdir("data")
        shutil.move("OUTCAR", f"data/OUTCAR_{idx}")

    def _apply_convergence_strategy(self, attempt):
        """
        Attempt 2: Mixing parameter
        Attempt 3: ALGO = Normal
        Attempt 4: ALGO = All
        """
        print(f"Applying convergence strategy for attempt {attempt}...")
        shutil.copy2("INCAR_step_start", "INCAR")

        with open("INCAR", "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if "ALGO" in line or "AMIX" in line or "BMIX" in line:
                continue
            new_lines.append(line)

        if attempt == 2:
            print(" -> Strategy: Conservative Mixing")
            new_lines.append("ALGO = Fast\n")
            new_lines.append("AMIX = 0.2\n")
            new_lines.append("BMIX = 0.0001\n")

        elif attempt == 3:
            print(" -> Strategy: ALGO = Normal")
            new_lines.append("ALGO = Normal\n")
            new_lines.append("AMIX = 0.4\n")
            new_lines.append("BMIX = 1.0\n")

        else:
            print(" -> Strategy: ALGO = All")
            new_lines.append("ALGO = All\n")

        with open("INCAR", "w") as f:
            f.writelines(new_lines) 

# ==================
# Main MDI Driver
# ==================
def main(args):

    try:
        mdi.MDI_Init(args.mdi)
    except Exception as e:
        print(f"MDI Init failed: {e}")
        sys.exit(1)

    comm = mdi.MDI_Accept_Communicator()

    vasp = VASPCalculator(
        vasp_cmd=args.vasp_cmd,
        poscar_template=args.poscar_template,
        max_try=args.max_try
    )

    mdi.MDI_Send_Command("<NATOMS", comm)
    natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command("<NTYPES", comm)
    ntypes = mdi.MDI_Recv(1, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command("<TYPES", comm)
    atom_types = mdi.MDI_Recv(natoms, mdi.MDI_INT, comm)

    print(f"Driver Connected. AToms: {natoms}, Types: {ntypes}")
    print(f"VASP Command: {args.vasp_cmd}")

    for step in range(args.steps):
        mdi.MDI_Send_Command("<COORDS", comm)
        coords = mdi.MDI_Recv(natoms * 3, mdi.MDI_DOUBLE, comm)
        mdi.MDI_Send_Command("<CELL", comm)
        cell = mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)

        energy, forces, virial = vasp.run_calculation(coords, cell, atom_types)

        mdi.MDI_Send_Command(">FORCES", comm)
        mdi.MDI_Send(forces, natoms * 3, mdi.MDI_DOUBLE, comm)
        mdi.MDI_Send_Command(">ENERGY", comm)
        mdi.MDI_Send([energy], 1, mdi.MDI_DOUBLE, comm)
        mdi.MDI_Send_Command("TRAJ", comm)

        print(f"Step {step+1}/{args.steps} Completed. E={energy:.4f}")

        if (step + 1) & args.restart_freq == 0:
            cmd = f"write_restart restart.{step+1}.bin"
            mdi.MDI_Send_Command("<COMMAND", comm)
            mdi.MDI_Send(cmd, len(cmd), mdi.MDI_CHAR, comm)

    mdi.MDI_Send_Command("EXIT", comm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDI Python Driver for G-metaD")
    parser.add_argument("--mdi", required=True, type=str, help="MDI connection string")
    parser.add_argument("--vasp_cmd", type=str, default="mpirun -np 32 vasp_std")
    parser.add_argument("--steps", type=int, default=1000, help="MD steps")
    parser.add_argument("--restart_freq", type=int, default=100, help="Restart frequency")
    parser.add_argument("--poscar_template", type=str, default="POSCAR_template", help="POSCAR template path")
    parser.add_argument("--max_try", type=int, default=4, help="Max VASP retries per step")

    args = parser.parse_args()
    main(args)
