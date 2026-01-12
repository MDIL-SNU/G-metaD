import sys
import os
import shutil
import subprocess
import time
import argparse
import xml.etree.ElementTree as ET
from glob import glob
import mdi


BOHR_TO_ANG = 0.52917721067
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
EV_TO_HARTREE = 0.03674932248
HARTREE_TO_EV = 1.0 / EV_TO_HARTREE
KBAR_TO_EV_PER_ANG3 = 0.00062415091

class MDICalculator:
    def __init__(self, poscar_template):
        self.poscar_template = poscar_template
        self.elements = []
        self._parse_template()

    def _parse_template(self):
        if not os.path.exists(self.poscar_template):
            print(f"Error: {self.poscar_template} not found.")
            sys.exit(1)
        with open(self.poscar_template, "r") as f:
            lines = f.readlines()

        try:
            self.elements = lines[5].split()
            self.natoms_per_type = [int(x) for x in lines[6].split()]
            self.ntypes = len(self.natoms_per_type)
            self.natoms = sum(self.natoms_per_type)
        except IndexError:
            print("Error: POSCAR_template format invalid.")
            sys.exit(1)

    def run_calculation(self, coords, box, types):
        """
        Input: coords(Angstrom), box(Angstrom)
        Output: energy(eV), forces(eV/A), virial(eV/A^3)
        """
        raise NotImplementedError


class VASPCalculator(MDICalculator):
    def __init__(self, vasp_cmd, poscar_template, max_try=4):
        super().__init__(poscar_template)
        self.vasp_cmd = vasp_cmd
        self.max_try = max_try

    def write_poscar(self, coords, box, types):
        with open(self.poscar_template, "r") as f:
            template = f.readlines()
        with open("POSCAR", "w") as f:
            f.write(template[0])
            f.write(template[1])
            f.write(f" {box[0]:12.8f} {box[1]:12.8f} {box[2]:12.8f}\n")
            f.write(f" {box[3]:12.8f} {box[4]:12.8f} {box[5]:12.8f}\n")
            f.write(f" {box[6]:12.8f} {box[7]:12.8f} {box[8]:12.8f}\n")
            f.write(template[5])
            f.write(template[6])
            f.write("Cartesian\n")

            current_types = types if types else []

            if not current_types:
                current_types = []
                for type_idx, count in enumerate(self.natoms_per_type):
                    current_types.extend([type_idx + 1] * count)

            if len(current_types) != self.natoms:
                print(f"Error: Type mismatch! Expected {self.natoms}, got {len(current_types)}", flush=True)
                sys.exit(1)

            atom_data = []
            for i in range(self.natoms):
                t = current_types[i]
                atom_data.append({
                    'type': t,
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
            if not os.path.exists("vasprun.xml"):
                return 0, [], [], False
            tree = ET.parse("vasprun.xml")
            root = tree.getroot()
            calcs = root.findall("calculation")
            if not calcs:
                return 0, [], [], False
            last_calc = calcs[-1]
            scsteps = last_calc.findall("scstep")
            converged = len(scsteps) < 60

            # Energy
            e_free_elem = last_calc.find("energy/i[@name='e_fr_energy']")
            if e_free_elem is None:
                return 0, [], [], False
            e_free = float(e_free_elem.text)

            # Forces & stress
            forces = []
            tmp_stress = []
            stress = [0.0] * 9

            varrays = last_calc.findall("varray")
            for v in varrays:
                if v.attrib["name"] == "forces":
                    for line in v.findall("v"):
                        forces.extend([float(x) for x in line.text.split()])
                if v.attrib["name"] == "stress":
                    for line in v.findall("v"):
                        tmp_stress.append([float(x) for x in line.text.split()])
            
            s = tmp_stress
            stress[0] = s[0][0] * KBAR_TO_EV_PER_ANG3
            stress[4] = s[1][1] * KBAR_TO_EV_PER_ANG3
            stress[8] = s[2][2] * KBAR_TO_EV_PER_ANG3
            stress[1] = s[0][1] * KBAR_TO_EV_PER_ANG3
            stress[5] = s[1][2] * KBAR_TO_EV_PER_ANG3
            stress[2] = s[2][0] * KBAR_TO_EV_PER_ANG3
            stress[3] = s[0][1] * KBAR_TO_EV_PER_ANG3
            stress[7] = s[1][2] * KBAR_TO_EV_PER_ANG3
            stress[6] = s[2][0] * KBAR_TO_EV_PER_ANG3

            return e_free, forces, stress, converged

        except Exception as e:
            print(f"Error reading vasprun.xml: {e}", flush=True)
            return 0, [], [], False

    def _backup_outcar(self):
        outcars = glob("data/OUTCAR_*")
        idx = len(outcars)
        if not os.path.isdir("data"):
            os.mkdir("data")
        if os.path.exists("OUTCAR"):
            shutil.move("OUTCAR", f"data/OUTCAR_{idx}")

    def _apply_convergence_strategy(self, attempt):
        print(f"Applying convergence strategy for attempt {attempt}...", flush=True)
        shutil.copy2("INCAR_step_start", "INCAR")

        with open("INCAR", "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if "ALGO" in line or "AMIX" in line or "BMIX" in line:
                continue
            new_lines.append(line)

        if attempt == 2:
            new_lines.append("ALGO = Normal\n")

        elif attempt == 3:
            new_lines.append("ALGO = Normal\n")
            new_lines.append("AMIX = 0.2\n")
            new_lines.append("BMIX = 0.0001\n")

        else:
            new_lines.append("ALGO = All\n")

        with open("INCAR", "w") as f:
            f.writelines(new_lines) 

    def run_calculation(self, coords, box, types):
        self.write_poscar(coords, box, types)

        if not os.path.exists("INCAR_step_start"):
            if os.path.exists("INCAR"):
                shutil.copy2("INCAR", "INCAR_step_start")
            else:
                print("Error: INCAR file not found!", flush=True)
                sys.exit(1)

        ntry = 0
        while True:
            ntry += 1
            try:
                subprocess.check_output(self.vasp_cmd, stderr=subprocess.STDOUT, shell=True)
                
                energy, forces_sorted, virial, converged = self.read_vasprun()

                if converged:
                    self._backup_outcar()
                    shutil.copy2("INCAR_step_start", "INCAR")
                    
                    atom_indices = list(range(self.natoms))
                    current_types = types if types else []
                    if not current_types:
                        current_types = []
                        for type_idx, count in enumerate(self.natoms_per_type):
                            current_types.extend([type_idx + 1] * count)

                    sorted_indices = sorted(atom_indices, key=lambda k: current_types[k])
                    forces_ordered = [0.0] * (self.natoms * 3)

                    for i, original_idx in enumerate(sorted_indices):
                        if (3*i + 2) < len(forces_sorted):
                            forces_ordered[3 * original_idx + 0] = forces_sorted[3 * i + 0]
                            forces_ordered[3 * original_idx + 1] = forces_sorted[3 * i + 1]
                            forces_ordered[3 * original_idx + 2] = forces_sorted[3 * i + 2]

                    return energy, forces_ordered, virial

                else:
                    print(f"VASP Warning: Convergence failed (Attempt {ntry}/{self.max_try})", flush=True)
                    if ntry >= self.max_try:
                        print("VASP Error: All convergence strategies failed.", flush=True)
                        sys.exit(1)

                    self._apply_convergence_strategy(ntry + 1)

            except subprocess.CalledProcessError as e:
                print(f"VASP Execution Error (Attempt {ntry}): Return Code {e.returncode}", flush=True)
                print(f"VASP Output:\n{e.output.decode('utf-8')}", flush=True)
                
                if ntry >= self.max_try:
                    sys.exit(1)
                time.sleep(10)


class MLFFCalculator(MDICalculator):
    def __init__(self, model_path, poscar_template):
        super().__init__(poscar_template)
        try:
            import numpy as np
            from ase import Atoms
            # Example: Sevennet 
            from sevenn.calculator import SevenNetCalculator
            # 'model_path' can be used like
            # self.calc = Calculator(model_path)
            self.np = np
            self.Atoms = Atoms
            self.calc = SevenNetCalculator(model='7net-omni', modal='mpa')
        except ImportError as e:
            print(f"Error in MLFFCalculator: {e}")
        
    def run_calculation(self, coords, box, types):
        cell = self.np.array(box).reshape(3, 3)
        pos = self.np.array(coords).reshape(-1, 3)
        symbols = [self.elements[t-1] for t in types]
        atoms = self.Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
        atoms.calc = self.calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().flatten().tolist()
        tmp_stress = atoms.get_stress(voigt=False)
        if tmp_stress.shape == (3, 3):
            stress = tmp_stress.flatten().tolist()
        else:
            s = tmp_stress
            stress = [s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]]

        return energy, forces, stress


def main(args):
    print("Engine Started. Connecting to Driver...", flush=True)
    try:
        mdi.MDI_Init(args.mdi)
        mdi.MDI_Register_Node("@DEFAULT")
        mdi.MDI_Register_Command("@DEFAULT", ">NATOMS")
        mdi.MDI_Register_Command("@DEFAULT", ">TYPES")
        mdi.MDI_Register_Command("@DEFAULT", "<STRESS")
        comm = mdi.MDI_Accept_Communicator()
        print("MDI Connected.", flush=True)

        if args.calculator == "vasp":
            calc = VASPCalculator(args.vasp_cmd, args.poscar_template, max_try=args.max_try)
        else:
            calc = MLFFCalculator(args.ml_model, args.poscar_template)

        natoms = calc.natoms
        coords = None
        cell = None
        atom_types = []

        energy = 0.0
        forces = [0.0] * (natoms * 3)
        virial = [0.0] * 6

        calc_needed = False

        while True:
            cmd = mdi.MDI_Recv_Command(comm)

            if cmd == ">NATOMS":
                natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)
            elif cmd == ">TYPES":
                atom_types = mdi.MDI_Recv(natoms, mdi.MDI_INT, comm)
            elif cmd == ">CELL":
                cell_mdi = mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)
                calc_needed = True
            elif cmd == ">COORDS":
                coords_mdi = mdi.MDI_Recv(natoms * 3, mdi.MDI_DOUBLE, comm)
                calc_needed = True

            elif cmd in ["<FORCES", "<ENERGY", "<STRESS"]:
                if calc_needed:
                    coords = [c * BOHR_TO_ANG for c in coords_mdi]
                    cell = [c * BOHR_TO_ANG for c in cell_mdi]

                    energy, forces, stress = calc.run_calculation(coords, cell, atom_types)

                    energy_mdi = energy * EV_TO_HARTREE
                    forces_mdi = [f * EV_TO_HARTREE * BOHR_TO_ANG for f in forces]
                    stress_mdi = [s * EV_TO_HARTREE * (BOHR_TO_ANG ** 3) for s in stress]
                    calc_needed = False

                if cmd == "<FORCES":
                    mdi.MDI_Send(forces_mdi, natoms * 3, mdi.MDI_DOUBLE, comm)
                elif cmd == "<ENERGY":
                    mdi.MDI_Send([energy_mdi], 1, mdi.MDI_DOUBLE, comm)
                elif cmd == "<STRESS":
                    mdi.MDI_Send(stress_mdi, 9, mdi.MDI_DOUBLE, comm)

            elif cmd == "EXIT":
                print("Engine Exit.", flush=True)
                break

            else:
                pass

    except Exception as e:
        print(f"Engine Error: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDI Python Driver for G-metaD")
    parser.add_argument("--mdi", required=True, type=str, help="MDI connection info")
    parser.add_argument("--poscar_template", type=str, default="POSCAR_template", help="Path to POSCAR template")
    parser.add_argument("--calculator", type=str, choices=['vasp', 'mlff'], default='vasp', help="Choose calculator backend")
    parser.add_argument("--vasp_cmd", type=str, default="mpirun -np 32 vasp_std")
    parser.add_argument("--ml_model", type=str, default="model.pt")
    parser.add_argument("--max_try", type=int, default=4, help="Max VASP retries per step")

    args = parser.parse_args()
    main(args)
