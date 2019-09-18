#include "mpi_util.h"

int MPI_Bcast_lazy(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, int delay) {
  int me = 0;
  int nprocs = 1;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);
  // proc 0 send the signal to start broadcast.
  if (me == 0) {
    for (int i=1; i<nprocs; i++) MPI_Send(NULL, 0, MPI_INT, i, SIGNAL_TAG, comm);
  // other processes waits (sleeps) until the signal arrives.
  // proc 0 is late here while checking for server response.
  } else {
    int flag = 0;
    MPI_Status status;
    while (!flag) {
      MPI_Iprobe(0, SIGNAL_TAG, comm, &flag, &status);
      usleep(delay);
    }
    MPI_Recv(NULL, 0, MPI_INT, 0, SIGNAL_TAG, comm, &status);
  }
  // finally do broadcast as all processors are ready.
  MPI_Bcast(buffer, count, datatype, root, comm);
}
