#!/usr/bin/env python2

# ----------------------------------------------------------------------
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# http://lammps.sandia.gov, Sandia National Laboratories
# Steve Plimpton, sjplimp@sandia.gov
# modified by Dongsun Yoo, will1792@snu.ac.kr
# ----------------------------------------------------------------------

# Syntax: vasp_wrap.py file/zmq POSCARfile vaspcmd

# wrapper on VASP to act as server program using CSlib
#   receives message with list of coords from client
#   creates VASP inputs
#   invokes VASP to calculate self-consistent energy of that config
#   reads VASP outputs
#   sends message with energy, forces, pressure to client

# NOTES:
# check to insure basic VASP input files are in place?
# could archive VASP input/output in special filenames or dirs?
# need to check that POTCAR file is consistent with atom ordering?
# could make syntax for launching VASP more flexible
#   e.g. command-line arg for # of procs
# detect if VASP had an error and return ERROR field, e.g. non-convergence ??

from __future__ import print_function
import sys

version = sys.version_info[0]
if version == 3:
    sys.exit("The CSlib python wrapper does not yet support python 3")

import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from cslib import CSlib

MAX_TRY = 3

# enums matching FixClientMD class in LAMMPS
SETUP, STEP = range(1, 2 + 1)
DIM, PERIODICITY, ORIGIN, BOX, NATOMS, NTYPES, TYPES, COORDS, UNITS, CHARGE = range(
    1, 10 + 1
)
FORCES, ENERGY, VIRIAL, ERROR = range(1, 4 + 1)


def error(txt):
    """Print error message and exit.
    """
    print("ERROR:", txt)
    sys.exit(1)


def vasp_setup(poscar):
    """Read initial VASP POSCAR file to setup problem.
    Return natoms, ntypes, box, and natoms_per_type.
    """
    ps = open(poscar, "r").readlines()

    # box size
    words = ps[2].split()
    xbox = float(words[0])
    words = ps[3].split()
    ybox = float(words[1])
    words = ps[4].split()
    zbox = float(words[2])
    box = [xbox, ybox, zbox]

    natoms_per_type = []
    words = ps[6].split()
    for word in words:
        if word == "#":
            break
        natoms_per_type.append(int(word))
    ntypes = len(natoms_per_type)
    natoms = sum(natoms_per_type)

    return natoms, ntypes, box, natoms_per_type


def poscar_write(poscar, natoms, ntypes, types, coords, box):
    """Write a new POSCAR file for VASP.
    """
    psold = open(poscar, "r").readlines()
    psnew = open("POSCAR", "w")

    # header, including box size
    psnew.write(psold[0])
    psnew.write(psold[1])
    psnew.write("%g %g %g\n" % (box[0], box[1], box[2]))
    psnew.write("%g %g %g\n" % (box[3], box[4], box[5]))
    psnew.write("%g %g %g\n" % (box[6], box[7], box[8]))
    psnew.write(psold[5])
    psnew.write(psold[6])

    # per-atom coords
    # grouped by types
    psnew.write("Cartesian\n")

    for itype in range(1, ntypes + 1):
        for i in range(natoms):
            if types[i] != itype:
                continue
            x = coords[3 * i + 0]
            y = coords[3 * i + 1]
            z = coords[3 * i + 2]
            aline = "  %g %g %g\n" % (x, y, z)
            psnew.write(aline)

    psnew.close()


def vasprun_read():
    """Read a VASP output vasprun.xml file using ElementTree module.
    """
    tree = ET.parse("vasprun.xml")
    root = tree.getroot()

    energy = root.find("calculation/energy")
    for child in energy:
        if child.attrib["name"] == "e_fr_energy":
            eout = float(child.text)

    fout = []
    sout = []

    varrays = root.findall("calculation/varray")
    for varray in varrays:
        if varray.attrib["name"] == "forces":
            forces = varray.findall("v")
            for line in forces:
                fxyz = line.text.split()
                fxyz = [float(value) for value in fxyz]
                fout += fxyz
        if varray.attrib["name"] == "stress":
            tensor = varray.findall("v")
            stensor = []
            for line in tensor:
                sxyz = line.text.split()
                sxyz = [float(value) for value in sxyz]
                stensor.append(sxyz)
            sxx = stensor[0][0]
            syy = stensor[1][1]
            szz = stensor[2][2]
            # symmetrize off-diagonal components
            sxy = 0.5 * (stensor[0][1] + stensor[1][0])
            sxz = 0.5 * (stensor[0][2] + stensor[2][0])
            syz = 0.5 * (stensor[1][2] + stensor[2][1])
            sout = [sxx, syy, szz, sxy, sxz, syz]

    return eout, fout, sout


def main(argv):
    if len(argv) != 4:
        print("Syntax: python vasp_wrap.py file/zmq POSCARfile vaspcmd")
        sys.exit(1)

    mode = argv[1]
    poscar_template = argv[2]
    vaspcmd = argv[3]

    if mode == "file":
        cs = CSlib(1, mode, "tmp.couple", None)
    elif mode == "zmq":
        cs = CSlib(1, mode, "*:5555", None)
    else:
        print("Syntax: python vasp_wrap.py file/zmq POSCARfile vaspcmd")
        sys.exit(1)

    natoms, ntypes, box, natoms_per_type = vasp_setup(poscar_template)

    # initial message for MD protocol
    msgID, nfield, fieldID, fieldtype, fieldlen = cs.recv()
    if msgID != 0:
        error("Bad initial client/server handshake")
    protocol = cs.unpack_string(1)
    if protocol != "md":
        error("Mismatch in client/server protocol")
    cs.send(0, 0)

    # endless server loop
    while True:
        # recv message from client
        # msgID = 0 = all-done message
        msgID, nfield, fieldID, fieldtype, fieldlen = cs.recv()
        if msgID < 0:
            break

        # SETUP receive at beginning of each run
        # required fields: DIM, PERIODICTY, ORIGIN, BOX,
        #                  NATOMS, NTYPES, TYPES, COORDS
        # optional fields: others in enum above, but VASP ignores them
        if msgID == SETUP:
            origin = []
            box = []
            natoms_recv = ntypes_recv = 0
            types = []
            coords = []

            for field in fieldID:
                if field == DIM:
                    dim = cs.unpack_int(DIM)
                    if dim != 3:
                        error("VASP only performs 3d simulations")
                elif field == PERIODICITY:
                    periodicity = cs.unpack(PERIODICITY, 1)
                    if not periodicity[0] or not periodicity[1] or not periodicity[2]:
                        error(
                            "VASP wrapper only currently supports fully periodic systems"
                        )
                elif field == ORIGIN:
                    origin = cs.unpack(ORIGIN, 1)
                elif field == BOX:
                    box = cs.unpack(BOX, 1)
                elif field == NATOMS:
                    natoms_recv = cs.unpack_int(NATOMS)
                    if natoms != natoms_recv:
                        error("VASP wrapper mis-match in number of atoms")
                elif field == NTYPES:
                    ntypes_recv = cs.unpack_int(NTYPES)
                    if ntypes != ntypes_recv:
                        error("VASP wrapper mis-match in number of atom types")
                elif field == TYPES:
                    types = cs.unpack(TYPES, 1)
                elif field == COORDS:
                    coords = cs.unpack(COORDS, 1)

            if (
                not origin
                or not box
                or not natoms
                or not ntypes
                or not types
                or not coords
            ):
                error("Required VASP wrapper setup field not received")

            # count the number of each type and check the number matches
            ntypes_list_recv = []
            for itype in range(1, ntypes + 1):
                ntypes_list_recv.append(len([t for t in types if t == itype]))
            assert natoms_per_type == ntypes_list_recv
            offsets = np.cumsum([0] + natoms_per_type)[:-1]

        # STEP receive at each timestep of run or minimization
        # required fields: COORDS
        # optional fields: ORIGIN, BOX
        elif msgID == STEP:
            coords = []

            for field in fieldID:
                if field == COORDS:
                    coords = cs.unpack(COORDS, 1)
                elif field == ORIGIN:
                    origin = cs.unpack(ORIGIN, 1)
                elif field == BOX:
                    box = cs.unpack(BOX, 1)

            if not coords:
                error("Required VASP wrapper step field not received")

        else:
            error("VASP wrapper received unrecognized message")

        # create POSCAR file
        # atoms are sorted by type in this function
        poscar_write(poscar_template, natoms, ntypes, types, coords, box)

        # invoke VASP
        ntry = 0
	while True:
            try:
                ntry += 1
                subprocess.check_output(vaspcmd, stderr=subprocess.STDOUT, shell=True)
                break
            except subprocess.CalledProcessError:
                if ntry <= MAX_TRY:
                    print("CalledProcessError. Trying again... ({0:}/{1:})".format(ntry, MAX_TRY))
                    time.sleep(10)
                else:
                    print("Too many CalledProcessError. Aborting...")
                    sys.exit(1)

        # process VASP output
        energy, forces_tmp, virial = vasprun_read()

        # sort VASP forces by id (it was sorted by type)
        forces = []
        count = [0] * ntypes
        for t in types:
            i = offsets[t - 1] + count[t - 1]
            forces += [
                forces_tmp[3 * i + 0],
                forces_tmp[3 * i + 1],
                forces_tmp[3 * i + 2],
            ]
            count[t - 1] += 1

        # convert VASP kilobars to bars
        for i, value in enumerate(virial):
            virial[i] *= 1000.0

        # return forces, energy, pressure to client
        cs.send(msgID, 3)
        cs.pack(FORCES, 4, 3 * natoms, forces)
        cs.pack_double(ENERGY, energy)
        cs.pack(VIRIAL, 4, 6, virial)

    # final reply to client
    cs.send(0, 0)

    # clean-up
    del cs


if __name__ == "__main__":
    main(sys.argv)
