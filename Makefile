CC=gcc
MPICC=mpicc

# CFLAGS=-O2
CFLAGS=-O3 -flto
LDFLAGS=-flto
OMPFLAGS=-fopenmp

HOSTFILE ?= hosts.txt
NODES ?= 2
# --------- INCLUDE -------------
COMMON_INC=-I common
INC_SEQ=-I ml-kem-768
INC_OMP=-I ml-kem-768-omp

# --------- BINARI ---------------
SEQ=sequential_jobs/seq_mlkem
SEQ_OMP=sequential_jobs/seq_mlkem_omp
PIPE=pipeline/pipeline_mlkem
PIPE_MPI=pipeline/pipeline_mpi_mlkem

# --------- BUILD ----------------
all: seq seq_omp pipe pipe_mpi

seq:
	$(CC) $(CFLAGS) sequential_jobs/seq_mlkem.c report/report.c ml-kem-768/*.c common/*.c $(INC_SEQ) $(COMMON_INC) $(LDFLAGS) -o $(SEQ)

seq_omp:
	$(CC) $(CFLAGS) $(OMPFLAGS) sequential_jobs/seq_mlkem.c report/report.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) $(LDFLAGS) -o $(SEQ_OMP)

pipe:
	$(CC) $(CFLAGS) $(OMPFLAGS) pipeline/pipeline_mlkem.c report/report.c pipeline/pipeline.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) $(LDFLAGS) -o $(PIPE)

pipe_mpi:
	$(MPICC) $(CFLAGS) $(OMPFLAGS) pipeline/pipeline_mpi_mlkem.c report/report.c pipeline/pipeline.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) $(LDFLAGS) -o $(PIPE_MPI)

# -------------- RUN SEQUENTIAL ----------

run_seq:
	@for i in $$(seq 1 10); do \
		./$(SEQ); \
	done

run_seq_omp:
	@for i in $$(seq 1 10); do \
		OMP_NUM_THREADS=$(THREADS) ./$(SEQ_OMP); \
	done

run_all_seq_omp:
	@for omp in 2 3 4 5 8; do \
		for i in $$(seq 1 10); do \
			OMP_NUM_THREADS=$$omp \
			./$(SEQ_OMP); \
		done \
	done

run_pipe:
	@for i in $$(seq 1 10); do \
		OMP_NUM_THREADS=$(THREADS) ./$(PIPE); \
	done

run_all_pipe:
	@for omp in 2 3 4 5 8; do \
		for i in $$(seq 1 10); do \
			OMP_NUM_THREADS=$$omp \
			./$(PIPE); \
		done \
	done

run_mpi_local:
	@for i in $$(seq 1 10); do \
    	mpiexec -n $(NP) \
			-genv OMP_NUM_THREADS $(THREADS) \
			./$(PIPE_MPI); \
	done

run_all_mpi_local: 
# np >= 2, e np x omp <= 8 + oversubscription
	@for cfg in "2 2" "2 3" "2 4" "3 2" "3 3" "3 4" "4 2" "4 3" "4 4"; do \
		set -- $$cfg; \
		np=$$1; \
		omp=$$2; \
		for i in $$(seq 1 10); do \
			mpiexec -n $$np \
				-genv OMP_NUM_THREADS=$$omp \
				./$(PIPE_MPI); \
		done \
	done

run_mpi_cluster:
	@for i in $$(seq 1 10); do \
		mpiexec -f $(HOSTFILE) -n $(NP) -env OMP_NUM_THREADS $(THREADS) ./$(PIPE_MPI) $(NODES); \
	done

run_all_mpi_cluster:
# np >= 2, e np x omp <= 16 + oversubscription
	@for cfg in "2 2" "2 3" "2 4" "2 8" "3 2" "3 3" "3 4" "3 8" "4 2" "4 3" "4 4" "4 8" "8 2" "8 3" "8 4" "8 8"; do \
		set -- $$cfg; \
		np=$$1; \
		omp=$$2; \
		for i in $$(seq 1 10); do \
			mpiexec -f $(HOSTFILE) -n $$np \
				-genv OMP_NUM_THREADS=$$omp \
				./$(PIPE_MPI) $(NODES); \
		done \
	done

run_all_tests:
	$(MAKE) run_seq
	$(MAKE) run_all_seq_omp
	$(MAKE) run_all_pipe
	$(MAKE) run_all_mpi_local
	$(MAKE) run_all_mpi_cluster	

clean:
	rm -f $(SEQ) $(SEQ_OMP) $(PIPE) $(PIPE_MPI)