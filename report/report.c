#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "report.h"

void write_report(const char* dir, const char* filename, int omp_enabled, int mpi_ranks, int omp_threads, int nodes, int n_jobs, int success, double time_sec, double keygen_sec, double enc_sec, double dec_sec) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);
    FILE *fp = fopen(path, "a"); // Apro il file in modalità append
    if (fp == NULL) {
        printf("Error opening file for writing: %s\n", path);
        return;
    }
    fseek(fp, 0, SEEK_END); // Sposto il puntatore alla fine del file
    if (ftell(fp) == 0) { // Se il file è vuoto, scrivo l'intestazione
       fprintf(fp, "OMP_ENABLED,MPI_RANKS,OMP_THREADS,TOT_WORKERS,NODES,N_JOBS,SUCCESS,FAIL,TIME_SEC,THROUGHPUT_JS,KEYGEN_SEC, ENC_SEC, DEC_SEC\n");
    }
    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n", 
        omp_enabled,
        mpi_ranks,
        omp_threads,
        omp_threads*mpi_ranks,
        nodes,
        n_jobs,
        success,
        n_jobs - success,
        time_sec,
        (double)n_jobs / time_sec,
        keygen_sec,
        enc_sec,
        dec_sec
    );
    fclose(fp);
}