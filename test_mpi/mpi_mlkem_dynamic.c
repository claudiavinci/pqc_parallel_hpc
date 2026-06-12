#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "api.h"
#include <mpi.h>

#define N_JOBS 100000
#define CHUNK_SIZE 1000
// Tag MPI per distinguere i messaggi
#define TAG_REQUEST 1
#define TAG_JOB     2
#define TAG_DIE     3
// valori di ritorno per i KEM Job
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

int main(int argc, char *argv[]) {
    int rank, size;
    int provided;
    int requested = MPI_THREAD_FUNNELED;

    MPI_Init_thread(&argc, &argv, requested, &provided);

    if (provided < requested) {
        printf("MPI does not provide required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Master-Worker approach requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // ==================== LOGICA DEL MASTER ====================
    if (rank == 0) {
        int next_job = 0;
        int active_workers = size - 1;
        MPI_Status status;
        int dummy_buf;
        double global_elapsed = 0.0;

        while (active_workers > 0) {
            // Aspetta una richiesta da QUALSIASI worker
            MPI_Recv(&dummy_buf, 1, MPI_INT, MPI_ANY_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, &status);
            int worker_rank = status.MPI_SOURCE;

            if (next_job < N_JOBS) {
                // Ci sono ancora job disponibili: assegna il job al worker
                MPI_Send(&next_job, 1, MPI_INT, worker_rank, TAG_JOB, MPI_COMM_WORLD);
                next_job += CHUNK_SIZE;
            } else {
                // Job terminati: invia il segnale di terminazione al worker
                MPI_Send(&dummy_buf, 0, MPI_INT, worker_rank, TAG_DIE, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        double t1 = MPI_Wtime();
        global_elapsed = t1 - t0;
        printf("\n=========== PARALLEL EXECUTION COMPLETED ===========");
        printf("\nMapping:          DYNAMIC (Chunking)");
        printf("\nActive MPI nodes:  %d", size);
        printf("\nJobs requested:    %d", N_JOBS);
        printf("\nTotal time:    %f sec", global_elapsed);
        printf("\n====================================================\n");
    } 
    // ==================== LOGICA DEI WORKER ====================
    else {
        int job_to_do;
        int dummy_req = 1;
        MPI_Status status;
        kem_job current_job;

        while (1) {
            // 1. Invia una richiesta di lavoro al Master
            MPI_Send(&dummy_req, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

            // 2. Ricevi la risposta dal Master (può essere un JOB o un segnale di DIE)
            MPI_Recv(&job_to_do, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DIE) {
                // Non c'è più lavoro, esci dal ciclo
                break;
            }

            int curr_chunk_size = CHUNK_SIZE;
            if (job_to_do + CHUNK_SIZE > N_JOBS) {
                curr_chunk_size = N_JOBS - job_to_do; // Adatta l'ultimo se supera N_JOBS
            }
            for (int i = 0; i < curr_chunk_size; i++) {
                // 3. Esegui il KEM Job (Sfrutta OpenMP internamente)
                int status = run_kem_job(&current_job);
            }
        }
    }

    MPI_Finalize();
    return 0;
}