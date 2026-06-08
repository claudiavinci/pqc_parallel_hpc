#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "report.h"

void write_report(const char* dir, const char* filename, int omp_enabled, int n_threads, int n_jobs, int success, double time_sec) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);
    FILE *fp = fopen(path, "a"); // Apro il file in modalità append
    if (fp == NULL) {
        printf("Error opening file for writing: %s\n", path);
        return;
    }
    fseek(fp, 0, SEEK_END); // Sposto il puntatore alla fine del file
    if (ftell(fp) == 0) { // Se il file è vuoto, scrivo l'intestazione
       fprintf(fp, "OMP_ENABLED,N_THREADS,N_JOBS,SUCCESS,FAIL,TIME_SEC,THROUGHPUT_JS\n");
    }
    fprintf(fp, "%d,%d,%d,%d,%d,%f,%f\n", 
        omp_enabled,
        n_threads,
        n_jobs,
        success,
        n_jobs - success,
        time_sec,
        (double)n_jobs / time_sec
    );
    fclose(fp);
}