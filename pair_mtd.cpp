/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_mtd.h"
#include "symmetry_functions.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "pointers.h"
#include "update.h"
#include "utils.h"
#include "Eigen/Dense"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace Eigen;

#define MAXLINE 50000

/* ---------------------------------------------------------------------- */
// Constructor

PairMTD::PairMTD(LAMMPS *lmp) : Pair(lmp) {
  nelements = 0;
  elements = NULL;
  map = NULL;
  // manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */
// Destructor

PairMTD::~PairMTD() {
  if (elements)
    for (int i = 0; i < nelements; i++) delete[] elements[i];
  delete[] elements;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete[] map;
  }

  for (int i = 0; i < nelements; i++) free_bias(biases[i]);
  delete[] biases;
  delete[] height;
  delete[] sigma_sq;
  delete[] nevery;
  delete[] bias_coeff;
}

/* ---------------------------------------------------------------------- */

void PairMTD::compute(int eflag, int vflag) {
  int i, ip, j, jp, k, kp, n, np, ii, jj, kk, tt, nn, inum, jnum;
  int itype, jtype, ktype, ielem, jelem, kelem;
  double xtmp, ytmp, ztmp, evdwl, fpair, dradtmp, tmpE;
  double dangtmp[3];
  double tmpd[9];
  double precal[17];
  // precal: cfij, dcfij, cfik, dcfik, cfjk, dcfjk, dist_square_sum,
  //         cosval, dcosval/dij, dcosval/dik, dcosval/djk
  double delij[3], delik[3], deljk[3], vecij[3], vecik[3], vecjk[3];
  double Rij, Rik, Rjk, rRij, rRik, rRjk, cutij, cutik, cutjk;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  struct Symc *sym;
  int tot_at = atom->natoms;
  int nsym;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    ip = tag[i] - 1;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    ielem = map[itype];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;
    nsym = biases[ielem].nsym;

    // Check for not implemented symfunc type.
    for (tt = 0; tt < nsym; tt++) {
      bool implemented = false;
      sym = &biases[ielem].slists[tt];
      for (int i = 0;
           i < sizeof(IMPLEMENTED_TYPE) / sizeof(IMPLEMENTED_TYPE[0]); i++) {
        if ((sym->stype) == IMPLEMENTED_TYPE[i]) {
          implemented = true;
          break;
        }
      }
      if (!implemented)
        error->all(FLERR, "Not implemented symmetry function type!");
    }

    double *symvec = new double[nsym]();
    double *dsymvec = new double[nsym]();
    double *tmpf = new double[nsym * (jnum + 1) * 3]();

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      // j &= NEIGHMASK; // What is this?
      jp = tag[j] - 1;

      delij[0] = x[j][0] - xtmp;
      delij[1] = x[j][1] - ytmp;
      delij[2] = x[j][2] - ztmp;
      Rij = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
      jtype = type[j];
      jelem = map[jtype];

      if (Rij < 0.0001 || Rij > cutsq[itype][jtype]) {
        continue;
      }

      rRij = sqrt(Rij);
      vecij[0] = delij[0] / rRij;
      vecij[1] = delij[1] / rRij;
      vecij[2] = delij[2] / rRij;

      // calc radial symfunc
      for (tt = 0; tt < nsym; tt++) {
        sym = &biases[ielem].slists[tt];
        if (rRij > sym->coefs[0]) continue;
        if (sym->atype[0] != jelem) continue;
        if (sym->stype == 2) {
          cutf2(rRij, sym->coefs[0], precal[0], precal[1], 0);

          symvec[tt] += G2(rRij, precal, sym->coefs, dradtmp);
          tmpd[0] = dradtmp * vecij[0];
          tmpd[1] = dradtmp * vecij[1];
          tmpd[2] = dradtmp * vecij[2];

          tmpf[tt * (jnum + 1) * 3 + jj * 3 + 0] += tmpd[0];
          tmpf[tt * (jnum + 1) * 3 + jj * 3 + 1] += tmpd[1];
          tmpf[tt * (jnum + 1) * 3 + jj * 3 + 2] += tmpd[2];

          tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 0] -= tmpd[0];
          tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 1] -= tmpd[1];
          tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 2] -= tmpd[2];
        } else
          continue;
      }

      if (rRij > max_rc_ang) continue;
      for (kk = jj + 1; kk < jnum; kk++) {
        k = jlist[kk];
        // k &= NEIGHMASK;
        kp = tag[k] - 1;
        delik[0] = x[k][0] - xtmp;
        delik[1] = x[k][1] - ytmp;
        delik[2] = x[k][2] - ztmp;
        Rik = delik[0] * delik[0] + delik[1] * delik[1] + delik[2] * delik[2];
        if (Rik < 0.0001) continue;
        rRik = sqrt(Rik);
        if (rRik > max_rc_ang) continue;
        ktype = type[k];
        kelem = map[ktype];

        deljk[0] = x[k][0] - x[j][0];
        deljk[1] = x[k][1] - x[j][1];
        deljk[2] = x[k][2] - x[j][2];
        Rjk = deljk[0] * deljk[0] + deljk[1] * deljk[1] + deljk[2] * deljk[2];
        if (Rjk < 0.0001) continue;
        rRjk = sqrt(Rjk);

        vecik[0] = delik[0] / rRik;
        vecik[1] = delik[1] / rRik;
        vecik[2] = delik[2] / rRik;

        vecjk[0] = deljk[0] / rRjk;
        vecjk[1] = deljk[1] / rRjk;
        vecjk[2] = deljk[2] / rRjk;

        // Note that Rij = rRij * rRij.
        precal[7] = (Rij + Rik - Rjk) / 2 / rRij / rRik;
        // dcos(theta)/db = precal[8]
        // dcos(theta)/dc = precal[9]
        // dcos(theta)/da = precal[10]
        precal[8] = 0.5 * (1 / rRik + 1 / Rij * (Rjk / rRik - rRik));
        precal[9] = 0.5 * (1 / rRij + 1 / Rik * (Rjk / rRij - rRij));
        precal[10] = -rRjk / rRij / rRik;
        // Calculate precal[6] only when there is at least one G4.
        // (precal[6] is only used by G4.)
        if (nsf[4] > 0) {
          precal[6] = Rij + Rik + Rjk;
        }
        if (nsf[5] > 0) {
          precal[11] = Rij + Rik;
        }

        // calc angular symfunc
        for (tt = 0; tt < nsym; tt++) {
          sym = &biases[ielem].slists[tt];
          if (rRik > sym->coefs[0]) continue;
          if (!((sym->atype[0] == jelem && sym->atype[1] == kelem) ||
                (sym->atype[0] == kelem && sym->atype[1] == jelem)))
            continue;
          if ((sym->stype) == 4) {
            if (rRjk > sym->coefs[0]) continue;
            cutf2(rRij, biases[ielem].slists[tt].coefs[0], precal[0], precal[1], 0);
            cutf2(rRik, biases[ielem].slists[tt].coefs[0], precal[2], precal[3], 1);
            cutf2(rRjk, biases[ielem].slists[tt].coefs[0], precal[4], precal[5], 2);

            symvec[tt] += G4(rRij, rRik, rRjk, biases[ielem].powtwo[tt], precal,
                             sym->coefs, dangtmp, biases[ielem].powint[tt]);

            tmpd[0] = dangtmp[0] * vecij[0];
            tmpd[1] = dangtmp[0] * vecij[1];
            tmpd[2] = dangtmp[0] * vecij[2];
            tmpd[3] = dangtmp[1] * vecik[0];
            tmpd[4] = dangtmp[1] * vecik[1];
            tmpd[5] = dangtmp[1] * vecik[2];
            tmpd[6] = dangtmp[2] * vecjk[0];
            tmpd[7] = dangtmp[2] * vecjk[1];
            tmpd[8] = dangtmp[2] * vecjk[2];

            tmpf[tt * (jnum + 1) * 3 + jj * 3 + 0] += tmpd[0] - tmpd[6];
            tmpf[tt * (jnum + 1) * 3 + jj * 3 + 1] += tmpd[1] - tmpd[7];
            tmpf[tt * (jnum + 1) * 3 + jj * 3 + 2] += tmpd[2] - tmpd[8];

            tmpf[tt * (jnum + 1) * 3 + kk * 3 + 0] += tmpd[3] + tmpd[6];
            tmpf[tt * (jnum + 1) * 3 + kk * 3 + 1] += tmpd[4] + tmpd[7];
            tmpf[tt * (jnum + 1) * 3 + kk * 3 + 2] += tmpd[5] + tmpd[8];

            tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 0] -= tmpd[0] + tmpd[3];
            tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 1] -= tmpd[1] + tmpd[4];
            tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 2] -= tmpd[2] + tmpd[5];
          } else if ((sym->stype) == 5) {
            cutf2(rRij, biases[ielem].slists[tt].coefs[0], precal[0], precal[1], 0);
            cutf2(rRik, biases[ielem].slists[tt].coefs[0], precal[2], precal[3], 1);

            symvec[tt] += G5(rRij, rRik, biases[ielem].powtwo[tt], precal,
                             sym->coefs, dangtmp, biases[ielem].powint[tt]);

            tmpd[0] = dangtmp[0] * vecij[0];
            tmpd[1] = dangtmp[0] * vecij[1];
            tmpd[2] = dangtmp[0] * vecij[2];
            tmpd[3] = dangtmp[1] * vecik[0];
            tmpd[4] = dangtmp[1] * vecik[1];
            tmpd[5] = dangtmp[1] * vecik[2];
            tmpd[6] = dangtmp[2] * vecjk[0];
            tmpd[7] = dangtmp[2] * vecjk[1];
            tmpd[8] = dangtmp[2] * vecjk[2];

            tmpf[tt * (jnum + 1) * 3 + jj * 3 + 0] += tmpd[0] - tmpd[6];
            tmpf[tt * (jnum + 1) * 3 + jj * 3 + 1] += tmpd[1] - tmpd[7];
            tmpf[tt * (jnum + 1) * 3 + jj * 3 + 2] += tmpd[2] - tmpd[8];

            tmpf[tt * (jnum + 1) * 3 + kk * 3 + 0] += tmpd[3] + tmpd[6];
            tmpf[tt * (jnum + 1) * 3 + kk * 3 + 1] += tmpd[4] + tmpd[7];
            tmpf[tt * (jnum + 1) * 3 + kk * 3 + 2] += tmpd[5] + tmpd[8];

            tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 0] -= tmpd[0] + tmpd[3];
            tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 1] -= tmpd[1] + tmpd[4];
            tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 2] -= tmpd[2] + tmpd[5];
          } else
            continue;
        }
      }
    }

    tmpE = eval_bias(symvec, tmpf, jnum, dsymvec, biases[ielem], itype, ielem);
    // Add bias potential to Coulomb energy for monitoring purpose.
    if (eflag_global) {
      eng_coul += tmpE;
    }
    if (eflag_atom) {
      eatom[i] += tmpE;
    }

    // update force
    // Since we scaled tmpf, we do not need to scale dsymvec.
    // Delete scale process 2020.08.25
    for (tt = 0; tt < nsym; tt++) {
      for (nn = 0; nn < jnum; nn++) {
        n = jlist[nn];
        f[n][0] -= tmpf[tt * (jnum + 1) * 3 + nn * 3 + 0] * dsymvec[tt];
        f[n][1] -= tmpf[tt * (jnum + 1) * 3 + nn * 3 + 1] * dsymvec[tt];
        f[n][2] -= tmpf[tt * (jnum + 1) * 3 + nn * 3 + 2] * dsymvec[tt];
      }
      f[i][0] -= tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 0] * dsymvec[tt];
      f[i][1] -= tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 1] * dsymvec[tt];
      f[i][2] -= tmpf[tt * (jnum + 1) * 3 + jnum * 3 + 2] * dsymvec[tt];
    }

    delete[] symvec;
    delete[] dsymvec;
    delete[] tmpf;

  }

  // `tail_flag` is used to turn on/off adding bias.
  // `pair_modify tail no` (default): add bias.
  // `pair_modify tail yes`: do not add bias.
  if (!tail_flag) {
    for (int i = 0; i < nelements; i++) update_bias(biases[i]);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMTD::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  map = new int[n + 1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMTD::settings(int narg, char **arg) {
  reg = utils::numeric(FLERR, arg[0], false, lmp);
  nelements = utils::inumeric(FLERR, arg[1], false, lmp);
  height = new double[nelements];
  sigma_sq = new double[nelements];
  nevery = new int[nelements];
  elements = new char *[nelements];

  // Next (nelements * 4) arguments are per-element settings (height, sigma,
  // nevery).
  if (narg < 4 * nelements + 2) error->all(FLERR, "Illegal pair_style command");
  for (int i = 0; i < nelements; i++) {
    int n = strlen(arg[4 * i + 2]) + 1;
    elements[i] = new char[n];
    strcpy(elements[i], arg[4 * i + 2]);

    height[i] = utils::numeric(FLERR, arg[4 * i + 3], false, lmp);
    sigma_sq[i] = powint(utils::numeric(FLERR, arg[4 * i + 4], false, lmp), 2);
    nevery[i] = utils::inumeric(FLERR, arg[4 * i + 5], false, lmp);
    if (nevery[i] <= 0) error->all(FLERR, "Illegal pair_style command");
  }

  int o = 4 * nelements + 2;
  int ntypes = utils::inumeric(FLERR, arg[o], false, lmp);
  bias_coeff = new double[ntypes];

  // Next (ntypes) arguments are bias coefficients to each atom type.
  // Bias coefficient is multiplied to the bias potential.
  // Therefore, bias_coeff==0 means bias potential is turned off for that atom
  // type.
  if (narg < o + ntypes) error->all(FLERR, "Illegal pair_style command");
  for (int i = 0; i < ntypes; i++) {
    bias_coeff[i] = utils::numeric(FLERR, arg[o + i + 1], false, lmp);
    if (bias_coeff[i] < 0) error->all(FLERR, "Illegal pair_style command");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMTD::coeff(int narg, char **arg) {
  int i, j, n;

  if (narg != 3 + atom->ntypes)
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  // insure I,J args are * *

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i], "NULL") == 0) {
      map[i - 2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i], elements[j]) == 0) break;
    map[i - 2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j], arg[i]);
      nelements++;
    }
  }

  biases = new Bias[nelements + 1];  // extra one is used for reading irrelevant
                                     // elements.
  for (i = 0; i < nelements + 1; i++) {
    biases[i].num_data = 0;
  }

  // read potential file and initialize potential parameters
  read_file(arg[2]);

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++) setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

void PairMTD::read_file(char *fname) {
  int i, j;
  FILE *fp;
  if (comm->me == 0) {
    // read file
    fp = fopen(fname, "r");
    if (fp == NULL) {
      char str[128];
      sprintf(str, "Cannot open NN potential file %s", fname);
      error->one(FLERR, str);
    }
  }

  int n, nwords, nsym, isym;
  char line[MAXLINE], *ptr, *tstr;
  int eof = 0;
  int stats = 0;
  int nnet = nelements;
  int max_sym_line = 6;
  char **p_elem = new char *[nelements];
  int valid_count = 0;
  bool valid = false;
  cutmax = 0;
  max_rc_ang = 0.0;
  for (int i = 0; i < 5 + 1; i++) {
    nsf[i] = 0;
  }

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fp);
      if (ptr == NULL) {
        eof = 1;
        if (stats != 1) error->one(FLERR, "insufficient potential");
        fclose(fp);
      } else
        n = strlen(line) + 1;
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(&n, 1, MPI_INT, 0, world);
    MPI_Bcast(line, n, MPI_CHAR, 0, world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line, '#'))) *ptr = '\0';
    nwords = utils::count_words(line);
    if (nwords == 0) continue;

    // get all potential parameters
    if (stats == 0) {  // initialization
      // FIXME: p_elem usage?
      p_elem[0] = strtok(line, " \t\n\r\f");
      for (i = 1; i < nelements; i++) {
        p_elem[i] = strtok(NULL, " \t\n\r\f");
      }
      stats = 1;
    } else if (stats == 1) {  // potential element setting
      tstr = strtok(line, " \t\n\r\f");
      char *t_elem = strtok(NULL, " \t\n\r\f");
      double t_cut = atof(strtok(NULL, " \t\n\r\f"));
      if (t_cut > cutmax) cutmax = t_cut;
      nnet = nelements;
      for (i = 0; i < nelements; i++) {
        if (strcmp(t_elem, elements[i]) == 0) {
          nnet = i;
          break;
        }
      }
      if (nnet == nelements) {
        if (valid) {
          free_bias(biases[nnet]);
        }
        valid = true;
      } else {
        valid_count++;
      }
      stats = 2;
      // cutoff setting
      for (i = 1; i <= atom->ntypes; i++) {
        if (map[i] == nnet) {
          for (j = 1; j <= atom->ntypes; j++) {
            cutsq[i][j] = t_cut * t_cut;
          }
        }
      }
    } else if (stats == 2) {  // symfunc number setting
      tstr = strtok(line, " \t\n\r\f");
      if (strncmp(tstr, "SYM", 3) != 0)
        error->one(FLERR, "potential file error: missing info(# of symfunc)");
      nsym = atoi(strtok(NULL, " \t\n\r\f"));
      biases[nnet].nsym = nsym;
      biases[nnet].slists = new Symc[nsym]();
      stats = 3;
      isym = 0;
    } else if (stats == 3) {  // read symfunc parameters
      biases[nnet].slists[isym].stype = atoi(strtok(line, " \t\n\r\f"));
      biases[nnet].slists[isym].coefs[0] = atof(strtok(NULL, " \t\n\r\f"));
      biases[nnet].slists[isym].coefs[1] = atof(strtok(NULL, " \t\n\r\f"));
      biases[nnet].slists[isym].coefs[2] = atof(strtok(NULL, " \t\n\r\f"));
      biases[nnet].slists[isym].coefs[3] = atof(strtok(NULL, " \t\n\r\f"));

      tstr = strtok(NULL, " \t\n\r\f");
      biases[nnet].slists[isym].atype[0] = nelements;
      for (i = 0; i < nelements; i++) {
        if (strcmp(tstr, elements[i]) == 0) {
          biases[nnet].slists[isym].atype[0] = i;
          break;
        }
      }
      // In this code, SF type >= 4 means that it is angular function.
      if (biases[nnet].slists[isym].stype >= 4) {
        // Find maximum cutoff distance among angular functions.
        max_rc_ang = max(max_rc_ang, biases[nnet].slists[isym].coefs[0]);
        nsf[biases[nnet].slists[isym].stype] += 1;
        tstr = strtok(NULL, " \t\n\r\f");
        biases[nnet].slists[isym].atype[1] = nelements;
        for (i = 0; i < nelements; i++) {
          if (strcmp(tstr, elements[i]) == 0) {
            biases[nnet].slists[isym].atype[1] = i;
            break;
          }
        }
      }

      isym++;
      if (isym == nsym) {
        /* skip unnecessary information */
        if (comm->me == 0) {
          ptr = fgets(line, MAXLINE, fp);
          while (strcmp("\n", ptr) != 0) {
            ptr = fgets(line, MAXLINE, fp);
          }
        }
        stats = 1;
      }
    }
  }
  if (valid_count == 0)
    error->one(FLERR, "potential file error: invalid elements");

  delete[] p_elem;

  // pre-calculate some constants for symmetry functions.
  for (int i = 0; i < nelements; i++) {
    int nsym = biases[i].nsym;
    biases[i].powtwo = new double[nsym];
    biases[i].powint = new bool[nsym];
    for (int tt = 0; tt < nsym; tt++) {
      biases[i].powtwo[tt] = 0.0;
      biases[i].powint[tt] = false;

      if (biases[i].slists[tt].stype == 4 || biases[i].slists[tt].stype == 5) {
        if (biases[i].slists[tt].coefs[2] < 1.0)
          error->all(FLERR, "Zeta in G4/G5 must be greater or equal to 1.0!");
        biases[i].powtwo[tt] = pow(2, 1 - biases[i].slists[tt].coefs[2]);
        // powint indicates whether zeta is (almost) integer so that we can
        // treat it as integer and use pow_int.
        // This is used because pow_int is much faster than pow.
        biases[i].powint[tt] = (biases[i].slists[tt].coefs[2] -
                                int(biases[i].slists[tt].coefs[2])) < 1e-6;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMTD::init_style() {
  int irequest;

  irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMTD::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  return cutmax;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMTD::write_restart(FILE *fp) {
    write_restart_setting(fp);
    // Restart data format:
    // 1. Number of elements (int)
    // For each element:
    // 2. num_data (int)
    // 3. samples (vector<double>)

    if (comm->me == 0) {
        int nelem = nelements;
        fwrite(&nelem, sizeof(int), 1, fp);

        for (int i = 0; i < nelements; ++i) {
            int ndata = biases[i].num_data;
            fwrite(&ndata, sizeof(int), 1, fp);
            if (ndata >0) {
                fwrite(biases[i].samples.data(), sizeof(double), ndata, fp);
            }
        }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMTD::read_restart(FILE *fp) {
    read_restart_settings(fp);

    if (comm->me == 0) {
        int nelem_file;
        fread(&nelem_file, sizeof(int), 1, fp);
        if (nelem_file != nelements) {
            error->one(FLERR, "Number of elements in restart file does not match input script");
        }

        for (int i = 0; i < nelements; ++i) {
            int ndata;
            fread(&ndata, sizeof(int), 1, fp);

            biases[i].num_data = ndata;
            biases[i].samples.resize(ndata);

            if (ndata > 0) {
                fread(biases[i].samples.data(), sizeof(double), ndata, fp);
            }
        }
    }

    for (int i = 0; i < nelements; ++i) {
        MPI_Bcast(&biases[i].num_data, 1, MPI_INT, 0, world);

        if (comm->me != 0) {
            biases[i].samples.resize(biases[i].num_data);
        }

        if (biases[i].num_data > 0) {
            MPI_Bcast(biases[i].samples.data(), biases[i].num_data, MPI_DOUBLE, 0, world);
        }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMTD::write_restart_settings(FILE *fp) {
    if (comm->me == 0) {
        fwrite(&cutmax, sizeof(double), 1, fp);
        fwrite(&reg, sizeof(double), 1, fp);
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMTD::read_restart_settings(FILE *fp) {
    if (comm->me == 0) {
        fread(&cutmax, sizeof(double), 1, fp);
        fread(&reg, sizeof(double), 1, fp);
    }
    MPI_Bcast(&cutmax, 1, MPI_DOUBLE, 0, world);
    MPI_Bcast(&reg, 1, MPI_DOUBLE, 0, world);
}

/* ---------------------------------------------------------------------- */

double PairMTD::single(int i, int j, int itype, int jtype, double rsq,
                       double factor_coul, double factor_lj, double &fforce) {
  if (comm->me == 0) printf("single run\n");
  return factor_lj;
}

/* ----------------------------------------------------------------------
   free a Bias struct
------------------------------------------------------------------------- */

void PairMTD::free_bias(Bias &bias) {
  delete[] bias.slists;
  delete[] bias.powtwo;
  delete[] bias.powint;
}

/* ----------------------------------------------------------------------
   evaluate bias potential U(G)
------------------------------------------------------------------------- */

double PairMTD::eval_bias(const double *G, const double *dG_dR, int jnum,
                          double *dU_dG, Bias &bias, int itype, int ielem) {
  // Return the energy (and calculate derivatives) from bias potential.
  // dU/dG is written on double* dU_dG.
  // jnum+1 is the length of dG_dR.

  int nsym = bias.nsym;
  // The first nsym numbers are center of Gaussian,
  // the next one number is prefactor of Gaussian distribution,
  // and the next nsym * (nsym + 1) / 2 numbers are inverse of covariance matrix
  // (in lower-triangle form).
  // In total, there are `nsym * (nsym + 3) / 2 + 1` floats per sample.
  int ndata_per_sample = nsym * (nsym + 3) / 2 + 1;
  double U = 0.0;
  for (int i = 0; i < nsym; i++) {
    dU_dG[i] = 0.0;
  }

  // Evaluate only if coefficient is not zero.
  if (bias_coeff[itype - 1] > 0) {
    #pragma omp parallel
    {
      double* dU_dG_private = new double[nsym]{};
      // For each sample...
      #pragma omp for reduction(+:U)
      for (int i = 0; i < bias.num_data / ndata_per_sample; i++) {
        // U = prefactor * exp(-0.5 * xCx)
        double xCx = 0.0;
        for (int j = 0; j < nsym; j++) {
          for (int k = 0; k < j + 1; k++) {
            double s_inv_jk =
                bias.samples
                    [i * ndata_per_sample + nsym + 1 + j * (j + 1) / 2 + k];
            double sample_j = bias.samples[i * ndata_per_sample + j];
            double sample_k = bias.samples[i * ndata_per_sample + k];
            // Double off-diagonal components as we are computing
            // for only lower-triangle.
            if (j != k) s_inv_jk *= 2;
            xCx += (G[j] - sample_j) * s_inv_jk * (G[k] - sample_k);
          }
        }
        double prefactor =
             bias_coeff[itype - 1] * bias.samples[i * ndata_per_sample + nsym];
        // for relax
        //    - bias_coeff[itype - 1] * bias.samples[i * ndata_per_sample + nsym];
        double U_i = prefactor * exp(-0.5 * xCx);

        // Let U = A * exp(-0.5 * (x - mu)^T * Sigma^-1 * (x - mu))
        // (or U = A * exp(-0.5 * x^T * C * x)),
        // then dU/dx = -(x - mu)^T * Sigma^-1 * U
        // (or dU/dx = -x^T * C * U).
        for (int j = 0; j < nsym; j++) {
          double xC = 0.0;
          for (int k = 0; k < nsym; k++) {
            double sample_k = bias.samples[i * ndata_per_sample + k];
            // Since data is in lower-triangle form, j should be greater than or
            // equal to k.
            int jprime = max(j, k);
            int kprime = min(j, k);
            double s_inv_jk = bias.samples[i * ndata_per_sample + nsym + 1 +
                                           jprime * (jprime + 1) / 2 + kprime];
            xC += (G[k] - sample_k) * s_inv_jk;
          }
          dU_dG_private[j] += -xC * U_i;
        }
        U += U_i;
      }
      #pragma omp critical
      {
        for (int j = 0; j < nsym; j++) {
          dU_dG[j] += dU_dG_private[j];
        }
      }
      delete[] dU_dG_private;
    }
  }

  // Insert a new sample to the list of visited samples.
  // `tail_flag` is used to turn on/off adding bias.
  // `pair_modify tail no` (default): add bias.
  // `pair_modify tail yes`: do not add bias.
  if (update->ntimestep % nevery[ielem] == 0 && update->ntimestep != 0 && !tail_flag && bias_coeff[itype - 1] > 0)
    insert_sample(G, dG_dR, jnum, bias, ielem);

  return U;
}

/* ----------------------------------------------------------------------
   insert a new sample to the list of visited samples
------------------------------------------------------------------------- */

void PairMTD::insert_sample(const double *G, const double *dG_dR, int jnum,
                            Bias &bias, int ielem) {
  int nsym = bias.nsym;
  bias.samples.insert(bias.samples.end(), &G[0], &G[nsym]);
  // Construct convariance matrix.
  MatrixXd C(nsym, nsym);
  for (int i = 0; i < nsym; i++) {
    for (int j = 0; j < i + 1; j++) {
      double cov_ij = 0.0;
      // k={0,1,2} is {x,y,z} axis respectively.
      for (int k = 0; k < 3; k++) {
        // nn<jnum is dGi_dRj (j!=i) and nn=jnum is dGi_dRi.
        for (int nn = 0; nn < jnum + 1; nn++) {
          cov_ij += dG_dR[i * (jnum + 1) * 3 + nn * 3 + k] *
                    dG_dR[j * (jnum + 1) * 3 + nn * 3 + k];
        }
      }
      cov_ij *= sigma_sq[ielem];
      C(i, j) = cov_ij;
      C(j, i) = cov_ij;
    }
  }
  // Find the maximum eigenvalue.
  double max_eigval = 0.0;
  auto eigvals = C.eigenvalues();
  for (int i = 0; i < C.rows(); i++) {
    max_eigval = max(eigvals[i].real(), max_eigval);
  }
  // Add small number for numerical stability.
  // The value added depends on the maximum eigenvalue.
  // This is to make the covariance matrix have the same condition number,
  // preventing condition number being too high and unstable.
  for (int i = 0; i < nsym; i++) {
    C(i, i) += max_eigval * reg;
  }
  // Prefactor in Gaussian distribution (user-given height).
  // (ignore normalization constant 1/sqrt(det(C) * (2.0*M_PI)**D),
  // since then the values become so small in high-dimension.
  bias.samples.push_back(height[ielem]);

  MatrixXd Cinv = C.inverse();
  // Inverse of covariance matrix in lower-triangle form.
  for (int i = 0; i < nsym; i++) {
    for (int j = 0; j < i + 1; j++) {
      bias.samples.push_back(Cinv(i, j));
    }
  }
}

/* ----------------------------------------------------------------------
   update the number of visited samples
------------------------------------------------------------------------- */

void PairMTD::update_bias(Bias &bias) {
  // Exchange newly added samples between processors.
  int count =
      bias.samples.size() -
      bias.num_data;  // The number of data of newly added (local) samples.
  // Do not early-return when count==0, because a processor can have zero atom
  // assigned when system is heterogeneous.
  int *counts = new int[comm->nprocs];
  MPI_Allgather(&count, 1, MPI_INT, counts, 1, MPI_INT, world);
  int total_count = 0;
  for (int i = 0; i < comm->nprocs; i++) {
    total_count += counts[i];
  }
  if (total_count == 0) return;
  bias.samples.resize(bias.num_data + total_count);
  int *disps = new int[comm->nprocs];
  disps[0] = 0;
  for (int i = 1; i < comm->nprocs; i++) {
    disps[i] = disps[i - 1] + counts[i - 1];
  }
  // Copy my data to do in-place allgatherv.
  std::copy(&bias.samples[bias.num_data], &bias.samples[bias.num_data + count],
            &bias.samples[bias.num_data + disps[comm->me]]);
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                 &bias.samples[bias.num_data], counts, disps, MPI_DOUBLE,
                 world);
  // Update the number of data.
  bias.num_data = bias.num_data + total_count;

  delete[] counts;
  delete[] disps;
}
