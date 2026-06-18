#include "../common.h"
#include "../report/report.h"
#include <omp.h>
#include <string.h>
#include "pipeline.h"
#include <mpi.h>

#define N_THREADS omp_get_max_threads()
#define OMP_ENABLED 1

int main(int argc, char *argv[]) {

    printf("START MAIN\n");
    fflush(stdout);

    MPI_Init(&argc, &argv);
    
    printf("AFTER MPI_INIT\n");
    fflush(stdout);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_nodes = (argc > 1) ? atoi(argv[1]) : 1;

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int len;

    MPI_Get_processor_name(hostname, &len);
    printf("\nPipeline omp+mpi execution starting on Rank %d/%d running on %s", rank, size, hostname);
    fflush(stdout);

    int global_success = 0;
    int local_success = 0;

    static kem_job jobs[N_JOBS];

    int job_chunk = N_JOBS/size;
    int rem_jobs = N_JOBS % size;
    int start_job = rank * job_chunk + (rank < rem_jobs ? rank : rem_jobs);
    int end_job = start_job + job_chunk + (rank < rem_jobs ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime(); // Prendo il tempo di inizio

    run_pipeline_omp(jobs, &local_success, start_job, end_job);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double local_elapsed = t1 - t0;
    double global_elapsed = 0.0;

    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_success, &global_success, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0){
        write_report(REPORT_PATH,
                     "pipeline_mpi_results.csv",
                     OMP_ENABLED,
                     size,
                     N_THREADS,
                     n_nodes,
                     N_JOBS,
                     global_success,
                     global_elapsed
                    );

        printf("\n=== PIPELINE OMP+MPI EXECUTION COMPLETED ===\n");
    }
    MPI_Finalize();
    return 0;
}