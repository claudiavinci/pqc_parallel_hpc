#include "pipeline.h"
#include "api.h"
#include "../common.h"
#include <omp.h>
#include <string.h>


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

void run_pipeline_omp(kem_job *jobs, int *success, int start_job, int end_job, double *keygen_elapsed, double *enc_elapsed, double *dec_elapsed){

    #pragma omp parallel
    {
        #pragma omp single
        {   
            for(int i=start_job; i < end_job; i++){

                // ---------- KEYGEN STAGE ----------
                #pragma omp task firstprivate(i) depend(out: jobs[i].pk, jobs[i].sk)
                {
                    double t = omp_get_wtime();
                    keygen_stage(&jobs[i]);
                    double elapsed = omp_get_wtime() - t;
                    #pragma omp atomic
                    *keygen_elapsed += elapsed;                }

                // ---------- ENCAPSULATION STAGE ----------
                #pragma omp task firstprivate(i) depend(in: jobs[i].pk, jobs[i].sk) depend(out: jobs[i].ct, jobs[i].ss_enc)
                {
                    double t = omp_get_wtime();
                    enc_stage(&jobs[i]);
                    double elapsed = omp_get_wtime() - t;
                    #pragma omp atomic
                    *enc_elapsed += elapsed;
                }

                // ---------- DECAPSULATION STAGE ----------
                #pragma omp task firstprivate(i) depend(in: jobs[i].ct, jobs[i].ss_enc) depend(out: jobs[i].ss_dec)
                {
                    double t = omp_get_wtime();
                    dec_stage(&jobs[i]);
                    double elapsed = omp_get_wtime() - t;
                    #pragma omp atomic
                    *dec_elapsed += elapsed;
                }

                // ---------- CHECK STAGE ----------
                #pragma omp task firstprivate(i) depend(in: jobs[i].ss_dec)
                {
                    if (!check_stage(&jobs[i])){
                        // jobs[i].status = KEM_FAIL;
                        printf("Shared secrets do not match for job %d. Test failed.\n", i);
                    } else {
                        #pragma omp atomic
                        (*success)++;
                    }
                }
            }
        }
    }
}
