#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include "api.h"

#define N_JOBS 100000
// definisco le costanti di successo e fallimento (seguendo la convenzione di PQClean)
#define KEM_FAIL 1
#define KEM_SUCCESS 0

typedef struct {
    uint8_t pk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_SECRETKEYBYTES];
    uint8_t ct[PQCLEAN_MLKEM768_CLEAN_CRYPTO_CIPHERTEXTBYTES];
    uint8_t ss_enc[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
    uint8_t ss_dec[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
} kem_job;

int run_kem_job(kem_job *job){
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
    if(memcmp(job->ss_enc, job->ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0) {
        return KEM_SUCCESS;
    } else {
        printf("Shared secrets do not match. Test failed.\n");
        return KEM_FAIL;
    }
}

int main(int argc, char *argv[])
{
    int provided;
    int requested = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    
    if (provided < requested) {
        printf("MPI does not provide required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int job_chunk = N_JOBS/size;
    int rem_jobs = N_JOBS % size;
    int start_job = rank * job_chunk + (rank < rem_jobs ? rank : rem_jobs);
    int end_job = start_job + job_chunk + (rank < rem_jobs ? 1 : 0);
    // rem (remaining) è il “resto della divisione del lavoro” che viene distribuito 1 per processo ai primi rank.
    int local_success = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = start_job; i < end_job; i++) {
        kem_job job;
        int status = run_kem_job(&job);
        if (status == KEM_SUCCESS) {
            local_success++;
        }
    }

    double t1 = MPI_Wtime();
    double local_elapsed = t1-t0;
    double global_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_success = 0;
    MPI_Reduce(&local_success, &global_success, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=========== PARALLEL EXECUTION COMPLETED ===========");
        printf("\nMapping:          STATIC (Block Distribution)");
        printf("\nActive MPI nodes:  %d", size);
        printf("\nJobs requested:    %d", N_JOBS);
        printf("\nJobs succeded:    %d / %d", global_success, N_JOBS);
        printf("\nTotal time:    %f sec", global_elapsed);
        printf("\n====================================================\n");
    }

    MPI_Finalize();
    return 0;
}