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

int run_kem_job(kem_job *job) {
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(job->pk, job->sk);
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(job->ct, job->ss_enc, job->pk);
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(job->ss_dec, job->ct, job->sk);
    
    if (memcmp(job->ss_enc, job->ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0) {
        return KEM_SUCCESS;
    } else {
        printf("Shared secrets do not match. Test failed.\n");
        return KEM_FAIL;
    }

    // if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(job->pk, job->sk) != 0) {
    //     printf("Key pair generation failed\n");
    //     return KEM_FAIL;
    // }

    // // Encapsulation
    // if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(job->ct, job->ss_enc, job->pk) != 0) {
    //     printf("Encapsulation failed\n");
    //     return KEM_FAIL;
    // }

    // // Decapsulation
    // if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(job->ss_dec, job->ct, job->sk) != 0) {
    //     printf("Decapsulation failed\n");
    //     return KEM_FAIL;
    // }

    // Verify that the shared secrets match

}

int main(int argc, char *argv[]) {
    // Funzione standard C per prendere il tempo ad alta risoluzione (Monotonic clock)
    struct timespec start, end;

    int global_success = 0;
    struct timespec t0, t1;
    printf("\nSequential execution starting...");
    timespec_get(&t0, TIME_UTC); // Prendo il tempo di inizio
    // Ciclo for lineare: un solo core processa tutti i job consecutivamente
    // #pragma omp parallel for reduction(+:global_success) schedule(static)
    for (int i = 0; i < N_JOBS; i++) {
        kem_job job;
        
        int status = run_kem_job(&job);
        
        if (status == KEM_SUCCESS) {
            global_success++;
        }
    }
    timespec_get(&t1, TIME_UTC); // Prendo il tempo di fine
    double elapsed_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    
    // Stampa dei risultati

    write_report(REPORT_PATH, "seq_mlkem_results.csv", OMP_ENABLED, MPI_RANKS, N_THREADS, NODES, N_JOBS, global_success, elapsed_time);

    printf("\n========== SEQUENTIAL EXECUTION COMPLETED ==========\n");
    return 0;
}