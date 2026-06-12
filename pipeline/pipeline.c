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

void run_pipeline_omp(kem_job *jobs, int *success, int start_job, int end_job){
    #pragma omp parallel
    {
        #pragma omp single
        {
            for(int i=start_job; i < end_job; i++){
                jobs[i].status = KEM_SUCCESS;

                // ---------- KEYGEN STAGE ----------
                #pragma omp task firstprivate(i) depend(out: jobs[i].pk, jobs[i].sk)
                {
                    if (keygen_stage(&jobs[i]) != KEM_SUCCESS){
                        jobs[i].status = KEM_FAIL;
                        printf("Key pair generation failed for job %d\n", i);
                    }
                }

                // ---------- ENCAPSULATION STAGE ----------
                #pragma omp task firstprivate(i) depend(in: jobs[i].pk, jobs[i].sk) depend(out: jobs[i].ct, jobs[i].ss_enc)
                {
                    if (jobs[i].status == KEM_SUCCESS) {
                        if (enc_stage(&jobs[i]) != KEM_SUCCESS){
                            jobs[i].status = KEM_FAIL;
                            printf("Encapsulation failed for job %d\n", i);
                        }
                    }
                }

                // ---------- DECAPSULATION STAGE ----------
                #pragma omp task firstprivate(i) depend(in: jobs[i].ct, jobs[i].sk) depend(out: jobs[i].ss_dec)
                {
                    if (jobs[i].status == KEM_SUCCESS) {
                        if (dec_stage(&jobs[i]) != KEM_SUCCESS){
                            jobs[i].status = KEM_FAIL;
                            printf("Decapsulation failed for job %d\n", i);
                        }
                    }
                }

                // ---------- CHECK STAGE ----------
                #pragma omp task firstprivate(i) depend(in: jobs[i].ss_enc, jobs[i].ss_dec)
                {
                    if (jobs[i].status == KEM_SUCCESS){
                        if (!check_stage(&jobs[i])){
                            jobs[i].status = KEM_FAIL;
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
}