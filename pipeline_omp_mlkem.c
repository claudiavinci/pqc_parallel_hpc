#include "common.h"
#include <time.h>
#include "report.h"
#include <omp.h>
#include <string.h>
#include "pipeline.h"

#define N_THREADS omp_get_max_threads()
#define OMP_ENABLED 1
#define NODES 1
#define MPI_RANKS 1

int main(int argc, char *argv[]) {
    printf("\nPipeline execution starting...\n");

    int global_success = 0;
    struct timespec t0, t1;
    static kem_job jobs[N_JOBS];

    timespec_get(&t0, TIME_UTC); // Prendo il tempo di inizio

    run_pipeline_omp(jobs, &global_success, 0, N_JOBS);

    timespec_get(&t1, TIME_UTC); // Prendo il tempo di fine
    double elapsed_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    // Stampa dei risultati
    write_report("reports", "pipeline_omp_results.csv", OMP_ENABLED, MPI_RANKS, N_THREADS, NODES, N_JOBS, global_success, elapsed_time);
    printf("\n=== PIPELINE EXECUTION COMPLETED===\n");

    return 0;
}