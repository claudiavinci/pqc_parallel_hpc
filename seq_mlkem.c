#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h> // Libreria standard C per il calcolo del tempo
#include "api.h"
#include "report.h"

#ifdef _OPENMP
    #include <omp.h>
    #define N_THREADS omp_get_max_threads()
    #define OMP_ENABLED 1
#else
    #define N_THREADS 1
    #define OMP_ENABLED 0
#endif

#define N_JOBS 100000
// Definisco le costanti di successo e fallimento (seguendo la convenzione di PQClean)
#define KEM_FAIL 1
#define KEM_SUCCESS 0

typedef struct {
    uint8_t pk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_SECRETKEYBYTES];
    uint8_t ct[PQCLEAN_MLKEM768_CLEAN_CRYPTO_CIPHERTEXTBYTES];
    uint8_t ss_enc[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
    uint8_t ss_dec[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
} kem_job;

int run_kem_job(kem_job *job) {
    if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(job->pk, job->sk) != 0) {
        printf("Key pair generation failed\n");
        return KEM_FAIL;
    }

    // Encapsulation
    if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(job->ct, job->ss_enc, job->pk) != 0) {
        printf("Encapsulation failed\n");
        return KEM_FAIL;
    }

    // Decapsulation
    if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(job->ss_dec, job->ct, job->sk) != 0) {
        printf("Decapsulation failed\n");
        return KEM_FAIL;
    }

    // Verify that the shared secrets match
    if (memcmp(job->ss_enc, job->ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0) {
        return KEM_SUCCESS;
    } else {
        printf("Shared secrets do not match. Test failed.\n");
        return KEM_FAIL;
    }
}

int main(int argc, char *argv[]) {
    printf("Sequential execution starting...\n");

    // Funzione standard C per prendere il tempo ad alta risoluzione (Monotonic clock)
    struct timespec start, end;

    int global_success = 0;
    struct timespec t0, t1;
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
    double secondi = (double)(t1.tv_sec - t0.tv_sec);
    double nanosecondi = (double)(t1.tv_nsec - t0.tv_nsec) / 1000000000.0;
    double elapsed_time = secondi + nanosecondi;
    
    // Stampa dei risultati

    write_report("reports", "seq_mlkem_results.csv", OMP_ENABLED, N_THREADS, N_JOBS, global_success, elapsed_time);

    printf("========== SEQUENTIAL EXECUTION COMPLETED ==========\n");
    return 0;
}