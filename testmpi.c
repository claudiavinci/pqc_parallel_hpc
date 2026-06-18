#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    printf("before init\n");
    fflush(stdout);

    int err = MPI_Init(&argc, &argv);
    printf("MPI_Init returned %d\n", err);
    fflush(stdout);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("after init rank=%d\n", rank);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}