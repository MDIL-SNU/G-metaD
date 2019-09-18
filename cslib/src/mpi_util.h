#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#ifdef MPI_YES
#include <mpi.h>
#else
#include <mpi_dummy.h>
#endif
#include <unistd.h>
#include "msg.h"

#define SIGNAL_TAG 927

using namespace CSLIB_NS;

int MPI_Bcast_lazy(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, int delay);

#endif
