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

# For MLFF
try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    # Example: Sevennet 
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    pass

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
            print("Error: POSCAR_template format invalid. Ensure VASP 5.x format.")
            sys.exit(1)

    def run_calculation(self, coords, box, types):
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
            # Header from template
            f.write(template[0])
            f.write(template[1])
            f.write(f" {box[0]:12.8f} {box[1]:12.8f} {box[2]:12.8f}\n")
            f.write(f" {box[3]:12.8f} {box[4]:12.8f} {box[5]:12.8f}\n")
            f.write(f" {box[6]:12.8f} {box[7]:12.8f} {box[8]:12.8f}\n")
            f.write(template[5])
            f.write(template[6])
            f.write("Cartesian\n")

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
            virial = [0.0] * 6
            stress = []
            varrays = last_calc.findall("varray")
            for v in varrays:
                if v.attrib["name"] == "forces":
                    for line in v.findall("v"):
                        forces.extend([float(x) for x in line.text.split()])
                if v.attrib["name"] == "stress":
                    for line in v.findall("v"):
                        stress.append([float(x) for x in line.text.split()])
            
            if len(stress) >= 3:
                virial[0] = stress[0][0] * 1000.0
                virial[1] = stress[1][1] * 1000.0
                virial[2] = stress[2][2] * 1000.0
                virial[3] = 0.5 * (stress[0][1] + stress[1][0]) * 1000.0
                virial[4] = 0.5 * (stress[0][2] + stress[2][0]) * 1000.0
                virial[5] = 0.5 * (stress[1][2] + stress[2][1]) * 1000.0

            return e_free, forces, virial, converged

        except Exception as e:
            print(f"Error reading vasprun.xml: {e}", flush=True)
            return 0, [], [], False

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
                    sorted_indices = sorted(atom_indices, key=lambda k: types[k])
                    forces_ordered = [0.0] * (self.natoms * 3)
                    for i, original_idx in enumerate(sorted_indices):
                        if (3*i + 2) < len(forces_sorted): # 안전장치
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
            print(" -> Strategy: Conservative Mixing", flush=True)
            new_lines.append("ALGO = Fast\n")
            new_lines.append("AMIX = 0.2\n")
            new_lines.append("BMIX = 0.0001\n")

        elif attempt == 3:
            print(" -> Strategy: ALGO = Normal", flush=True)
            new_lines.append("ALGO = Normal\n")
            new_lines.append("AMIX = 0.4\n")
            new_lines.append("BMIX = 1.0\n")

        else:
            print(" -> Strategy: ALGO = All", flush=True)
            new_lines.append("ALGO = All\n")

        with open("INCAR", "w") as f:
            f.writelines(new_lines) 


class MLFFCalculator(MDICalculator):
    def __init__(self, model_path, poscar_template):
        super().__init__(poscar_template)
        self.calc = SevenNetCalculator(model='7net-omni', modal='mpa')
        
    def run_calculation(self, coords, box, types):
        cell = np.array(box).reshape(3, 3)
        pos = np.array(coords).reshape(-1, 3)
        symbols = [self.elements[t-1] for t in types]
        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
        atoms.calc = self.calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().flatten().tolist()
        stress = atoms.get_stress(voigt=False) * 160217.66
        virial = [
            stress[0, 0], stress[1, 1], stress[2, 2],
            stress[0, 1], stress[0, 2], stress[1, 2]
        ]
        return energy, forces, virial

def main(args):
    print("Driver Started. Connecting to MDI...", flush=True)
    try:
        try:
            mdi.MDI_Init(args.mdi)
        except Exception as e:
            print(f"MDI Init failed: {e}", flush=True)
            sys.exit(1)

        comm = mdi.MDI_Accept_Communicator()
        print("MDI Communicator Accepted.", flush=True)

        if args.calculator == "vasp":
            calc = VASPCalculator(args.vasp_cmd, args.poscar_template, max_try=args.max_try)
        else:
            calc = MLFFCalculator(args.ml_model, args.poscar_template)

        # Initial Info
        mdi.MDI_Send_Command("<NATOMS", comm)
        natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)
        
        mdi.MDI_Send_Command("<TYPES", comm)
        atom_types = mdi.MDI_Recv(natoms, mdi.MDI_INT, comm)
        ntypes = max(atom_types)

        mdi.MDI_Send_Command("@INIT_MD", comm)
        print(f"Driver Connected. Atoms: {natoms}, Types: {ntypes}", flush=True)

        for step in range(args.steps):
            mdi.MDI_Send_Command("<CELL", comm)
            cell = mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)
            mdi.MDI_Send_Command("@FORCES", comm)
            mdi.MDI_Send_Command("<COORDS", comm)
            coords = mdi.MDI_Recv(natoms * 3, mdi.MDI_DOUBLE, comm)

            energy, forces, virial = calc.run_calculation(coords, cell, atom_types)

            mdi.MDI_Send_Command(">FORCES", comm)
            mdi.MDI_Send(forces, natoms * 3, mdi.MDI_DOUBLE, comm)
            mdi.MDI_Send_Command(">ENERGY", comm)
            mdi.MDI_Send([energy], 1, mdi.MDI_DOUBLE, comm)
            mdi.MDI_Send_Command("@ENDSTEP", comm)

            print(f"Step {step+1}/{args.steps} E={energy:.4f}", flush=True)

            mdi.MDI_Send_Command("@DEFAULT", comm)
            if (step + 1) % args.restart_freq == 0:
                cmd = f"write_restart restart.{step+1}.bin"
                mdi.MDI_Send_Command("<COMMAND", comm)
                mdi.MDI_Send(cmd, len(cmd), mdi.MDI_CHAR, comm)

        mdi.MDI_Send_Command("EXIT", comm)

    except Exception as e:
        print(f"\nCRITICAL DRIVER ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()

        try:
            mdi.MDI_Send_Command("EXIT", comm)
        except:
            pass

        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDI Python Driver for G-metaD")
    parser.add_argument("--mdi", required=True, type=str, help="MDI connection info")
    parser.add_argument("--poscar_template", type=str, default="POSCAR_template", help="Path to POSCAR template")
    parser.add_argument("--calculator", type=str, choices=['vasp', 'mlff'], default='vasp', help="Choose calculator backend")
    parser.add_argument("--vasp_cmd", type=str, default="mpirun -np 32 vasp_std")
    parser.add_argument("--ml_model", type=str, default="model.pt")
    parser.add_argument("--steps", type=int, default=1000, help="MD steps")
    parser.add_argument("--restart_freq", type=int, default=100, help="Restart frequency")
    parser.add_argument("--max_try", type=int, default=4, help="Max VASP retries per step")

    args = parser.parse_args()
    main(args)
