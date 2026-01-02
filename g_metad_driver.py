import sys
import os
import shutil
import subprocess
import time
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import mdi


class VASPCalculator:
    def __init__(self, poscar_template="POSCAR_template", vasp_cmd="vasp_std"):
        self.poscar_template = poscar_template
        self.vasp_cmd = vasp_cmd
        self.max_try = 3
        self.natoms, self.ntypes, _, self.natoms_per_type = self.vasp_setup()

    def _vasp_setup(self):
        """Parse POSCAR_template to get atom info"""
        with open(self.poscar_template, "r") as f:
            lines = f.readlines()

        # Simple parsing for atom counts
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

        # Check convergence
        calcs = root.findall("calculation")
        if not calcs: return 0, [], [], False

        last_calc = calcs[-1]
        scsteps = last_calc.findall("scstep")
        converged = True
        if len(scsteps) >= 60:
            converged = False

        # Energy
        e_free = float(last_calc.find("energy/i[@name='e_fr_energy']").text)

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
        if len(stress) == 3:
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

        ntry = 0
        while True:
            ntry += 1
            try:
                subprocess.check_output(self.vasp_cmd, stderr=subprocess.STDOUT, shell=True)
                energy, forces_sorted_by_type, virial, converged = self.read_vasprun()

                if converged:
                    self._backup_outcar()
                    if ntry > 1:
                        shutil.copy2("INCAR_backup", INCAR)
                        break
                else:
                    print(f"VASP Warning: Step failed convergence (Attempt {ntry})")
                    if ntry >= self.max_try:
                        print("VASP Error: Max retries reached.")
                        sys.exit(1)

                    self._modify_incar_for_retry()

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

    def _modify_incar_for_retry(self):
        print("MOdifying INCAR for retry (Mixing parameters)...")
        shutil.copy2("INCAR", "INCAR_backup")
        with open("INCAR", "a") as f:
            f.write("\nAMIX = 0.2\nBMIX = 0.0001\n")
