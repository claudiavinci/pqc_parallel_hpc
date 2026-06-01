#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#include "api.h"

int main(int argc, char *argv[])
{
    uint8_t pk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_SECRETKEYBYTES];
    uint8_t ct[PQCLEAN_MLKEM768_CLEAN_CRYPTO_CIPHERTEXTBYTES];
    uint8_t ss_enc[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
    uint8_t ss_dec[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
    int N = 100000;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunk = N/size;
    int remaining = N % size;
    int start = rank * chunk + (rank < remaining ? rank : remaining);
    int end = start + chunk + (rank < remaining ? 1 : 0);
    // remaining è il “resto della divisione del lavoro” che viene distribuito 1 per processo ai primi rank.
    int ok = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = start; i < end; i++) {
        // Key pair generation
        if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(pk, sk) != 0) {
            printf("Key pair generation failed\n");
            continue;
        }

        // Encapsulation
        if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(ct, ss_enc, pk) != 0) {
            printf("Encapsulation failed\n");
            continue;
        }

        // Decapsulation
        if (PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(ss_dec, ct, sk) != 0) {
            printf("Decapsulation failed\n");
            continue;
        }

        // Verify that the shared secrets match
        if (memcmp(ss_enc, ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0) {
            // printf("Shared secrets match. Test passed.\n");
            ok++;
        } else {
            printf("Shared secrets do not match. Test failed.\n");
        }
        // if (i % 10000 == 0) {
        //     printf("Progress: %d\n", i);
        // }
    }
    // printf("Test results: %d/%d passed.\n", ok, 10000);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1-t0;
    double global_time;
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_ok;
    MPI_Reduce(&ok, &global_ok, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    
     if (rank == 0) {
        printf("Total time (MPI max): %f sec\n", global_time);
        printf("Total successes: %d / %d\n", global_ok, N);
        printf("Throughput: %f ops/sec\n", (double)N / global_time);
    }

    MPI_Finalize();
    return 0;
}