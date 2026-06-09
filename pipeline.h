#ifndef PIPELINE_H
#define PIPELINE_H

#include "common.h"

void run_pipeline_omp(kem_job *jobs, int *global_success, int start_job, int end_job);

#endif /* PIPELINE_H */