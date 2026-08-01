#define _POSIX_C_SOURCE 199309L

#include "../common.h"
#include "../report/report.h"
#include <time.h> // Libreria standard C per il calcolo del tempo
#include <string.h>


#define NODES 1
#define MPI_RANKS 1

#ifdef _OPENMP
    #include <omp.h>
    #define N_THREADS omp_get_max_threads()
    #define OMP_ENABLED 1
#else
    #define N_THREADS 1
    #define OMP_ENABLED 0
#endif

int run_kem_job(kem_job *job, double *keygen_sec, double *enc_sec, double *dec_sec) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0); // Prendo il tempo di inizio
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(job->pk, job->sk);
    clock_gettime(CLOCK_MONOTONIC, &t1); // Prendo il tempo di fine

    *keygen_sec += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &t0); // Prendo il tempo di inizio
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(job->ct, job->ss_enc, job->pk);
    clock_gettime(CLOCK_MONOTONIC, &t1); // Prendo il tempo di fine
    *enc_sec += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &t0); // Prendo il tempo di inizio
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(job->ss_dec, job->ct, job->sk);
    clock_gettime(CLOCK_MONOTONIC, &t1); // Prendo il tempo di fine
    *dec_sec += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    if (memcmp(job->ss_enc, job->ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0) {
        return KEM_SUCCESS;
    } else {
        printf("Shared secrets do not match. Test failed.\n");
        return KEM_FAIL;
    }
}

int main(int argc, char *argv[]) {
    // Funzione standard C per prendere il tempo ad alta risoluzione (Monotonic clock)
    printf("\nSequential execution starting...");

    int global_success = 0;
    double keygen_sec = 0.0;
    double enc_sec = 0.0;
    double dec_sec = 0.0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0); // Prendo il tempo di inizio

    for (int i = 0; i < N_JOBS; i++) {
        static kem_job job;
        
        int status = run_kem_job(&job, &keygen_sec, &enc_sec, &dec_sec);
        
        if (status == KEM_SUCCESS) {
            global_success++;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1); // Prendo il tempo di fine
    double elapsed_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    
    // Stampa dei risultati

    write_report(REPORT_PATH, "seq_mlkem_results.csv", OMP_ENABLED, MPI_RANKS, N_THREADS, NODES, N_JOBS, global_success, elapsed_time, keygen_sec, enc_sec, dec_sec);

    printf("\n========== SEQUENTIAL EXECUTION COMPLETED ==========\n");
    return 0;
}