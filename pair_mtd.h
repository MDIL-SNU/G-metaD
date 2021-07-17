/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mtd, PairMTD)

#else

#ifndef LMP_PAIR_MTD
#define LMP_PAIR_MTD
#define _USE_MATH_DEFINES

#include "pair.h"
#include "math.h"
#include <omp.h>
#include <iostream>
#include <vector>
using namespace std;

namespace LAMMPS_NS {

class PairMTD : public Pair {
 public:
  PairMTD(class LAMMPS *);
  virtual ~PairMTD();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);

 protected:
  // have to change
  struct Symc {
    int stype;        // symmetry function type
    double coefs[4];  // symmetry function coefficients
    int atype[2];     // related atom type(tmp)
  };

  // Bias is used to construct bias potential U(G) from visited samples.
  struct Bias {
    int nsym;        // The number of symmetry function components.
    int num_data;    // The number of data at the point of evaluation.
    Symc *slists;    // The parameters of symmetry functions.
    std::vector<double> samples;  // The samples that are visited. Used to
                                  // construct bias potential U(G).
    double *powtwo;
    bool *powint;
  };

  Bias *biases;     // One Bias per element.
  char **elements;  // names of unique elements
  int nelements;    // # of unique elements
  int *map;         // mapping from atom types to elements
  double cutmax;
  double max_rc_ang;
  int nsf[5 + 1];      // number of symmetry functions with type N.
  double *height;      // The height of gaussian bias function.
  double *sigma_sq;    // The square of sigma of gaussian bias function.
  int *nevery;         // The interval between each accumulation of density.
  double *bias_coeff;  // The bias coefficient multiplied to the bias potential.
  double reg;

  virtual void allocate();
  virtual void read_file(char *);
  void free_bias(Bias &);
  double eval_bias(const double *, const double *, int, double *, Bias &, int,
                   int);
  void insert_sample(const double *, const double *, int, Bias &, int);
  void update_bias(Bias &);
};
}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
