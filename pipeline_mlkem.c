#include "common.h"
#include <time.h>
#include "report.h"
#include <omp.h>
#include <string.h>

#define N_THREADS omp_get_max_threads()
#define OMP_ENABLED 1

// Pipeline stages definition
static inline int keygen_stage(kem_job *job){
    return PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(job->pk, job->sk);
}

static inline int enc_stage(kem_job *job){
    return PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(job->ct, job->ss_enc, job->pk);
}

static inline int dec_stage(kem_job *job){
    return PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(job->ss_dec, job->ct, job->sk);
}

static inline int check_stage(kem_job *job){
    return memcmp(job->ss_enc, job->ss_dec, PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES) == 0;
}

int main(int argc, char *argv[]) {
    printf("Pipeline execution starting...\n");

    int global_success = 0;
    struct timespec t0, t1;
    static kem_job jobs[N_JOBS];

    timespec_get(&t0, TIME_UTC); // Prendo il tempo di inizio

    #pragma omp parallel
    {
        #pragma omp single
        {
            for(int i=0; i < N_JOBS; i++){
                jobs[i].status = KEM_SUCCESS;

                // ---------- KEYGEN STAGE ----------
                #pragma omp task depend(out: jobs[i].pk, jobs[i].sk)
                {
                    if (keygen_stage(&jobs[i]) != KEM_SUCCESS){
                        jobs[i].status = KEM_FAIL;
                        printf("Key pair generation failed for job %d\n", i);
                    }
                }

                // ---------- ENCAPSULATION STAGE ----------
                #pragma omp task depend(in: jobs[i].pk, jobs[i].sk) depend(out: jobs[i].ct, jobs[i].ss_enc)
                {
                    if (jobs[i].status == KEM_SUCCESS) {
                        if (enc_stage(&jobs[i]) != KEM_SUCCESS){
                            jobs[i].status = KEM_FAIL;
                            printf("Encapsulation failed for job %d\n", i);
                        }
                    }
                }

                // ---------- DECAPSULATION STAGE ----------
                #pragma omp task depend(in: jobs[i].ct, jobs[i].sk) depend(out: jobs[i].ss_dec)
                {
                    if (jobs[i].status == KEM_SUCCESS) {
                        if (dec_stage(&jobs[i]) != KEM_SUCCESS){
                            jobs[i].status = KEM_FAIL;
                            printf("Decapsulation failed for job %d\n", i);
                        }
                    }
                }

                // ---------- CHECK STAGE ----------
                #pragma omp task depend(in: jobs[i].ss_enc, jobs[i].ss_dec)
                {
                    if (jobs[i].status == KEM_SUCCESS){
                        if (!check_stage(&jobs[i])){
                            jobs[i].status = KEM_FAIL;
                            printf("Shared secrets do not match for job %d. Test failed.\n", i);
                        } else {
                            #pragma omp atomic
                            global_success++;
                        }
                    }
                }
            }
        }
    }

    timespec_get(&t1, TIME_UTC); // Prendo il tempo di fine
    double elapsed_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    // Stampa dei risultati
    write_report("reports", "pipeline_mlkem_results.csv", OMP_ENABLED, N_THREADS, N_JOBS, global_success, elapsed_time);
    printf("\n=== PIPELINE EXECUTION COMPLETED===\n");

    return 0;
}