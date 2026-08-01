#define _POSIX_C_SOURCE 199309L

#include "../common.h"
#include <time.h>
#include "../report/report.h"
#include <omp.h>
#include <string.h>
#include "pipeline.h"

#define N_THREADS omp_get_max_threads()
#define OMP_ENABLED 1
#define NODES 1
#define MPI_RANKS 1

int main(int argc, char *argv[]) {
    printf("\nPipeline execution starting...");

    int global_success = 0;
    struct timespec t0, t1;
    static kem_job jobs[N_JOBS];
    static kem_timing timings[N_JOBS];
    double keygen_sec = 0.0; 
    double enc_sec = 0.0; 
    double dec_sec = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &t0); // Prendo il tempo di inizio

    run_pipeline_omp(jobs, &global_success, 0, N_JOBS, timings);

    clock_gettime(CLOCK_MONOTONIC, &t1); // Prendo il tempo di fine

    double elapsed_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    for (int i = 0; i < N_JOBS; i++) {

        keygen_sec += timings[i].keygen_time;
        enc_sec += timings[i].enc_time;
        dec_sec += timings[i].dec_time;

    }
    // Stampa dei risultati
    write_report(REPORT_PATH, "pipeline_results.csv", OMP_ENABLED, MPI_RANKS, N_THREADS, NODES, N_JOBS, global_success, elapsed_time, keygen_sec, enc_sec, dec_sec);
    printf("\n=== PIPELINE EXECUTION COMPLETED===\n");

    return 0;
}