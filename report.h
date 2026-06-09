#ifndef REPORT_H
#define REPORT_H

void write_report(const char* dir, const char* filename, int omp_enabled, int nodes, int n_threads, int n_jobs, int success, double time_sec);

#endif