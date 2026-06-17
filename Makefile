CC=gcc
MPICC=mpicc

CFLAGS=-O2
OMPFLAGS=-fopenmp

HOSTFILE ?= hosts.txt
NODES ?= 1

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
	$(CC) $(CFLAGS) sequential_jobs/seq_mlkem.c report/report.c ml-kem-768/*.c common/*.c $(INC_SEQ) $(COMMON_INC) -o $(SEQ)

seq_omp:
	$(CC) $(CFLAGS) $(OMPFLAGS) sequential_jobs/seq_mlkem.c report/report.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) -o $(SEQ_OMP)

pipe:
	$(CC) $(CFLAGS) $(OMPFLAGS) pipeline/pipeline_mlkem.c report/report.c pipeline/pipeline.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) -o $(PIPE)

pipe_mpi:
	$(MPICC) $(CFLAGS) $(OMPFLAGS) pipeline/pipeline_mpi_mlkem.c report/report.c pipeline/pipeline.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) -o $(PIPE_MPI)

# -------------- RUN SEQUENTIAL ----------

run_seq:
	for i in $$(seq 1 20); do \
		./$(SEQ); \
	done

run_seq_omp:
	for i in $$(seq 1 20); do \
		OMP_NUM_THREADS=$(THREADS) ./$(SEQ_OMP); \
	done

run_pipe:
	for i in $$(seq 1 20); do \
		OMP_NUM_THREADS=$(THREADS) ./$(PIPE); \
	done

run_mpi_local:
# 	for i in $$(seq 1 20); do \
		OMP_NUM_THREADS=$(THREADS) mpirun -np $(NP) ./$(PIPE_MPI); \
# 	done

	for i in $$(seq 1 20); do \
    	mpiexec -n $(NP) \
        	-env OMP_NUM_THREADS $(THREADS) \
           	./$(PIPE_MPI); \
	done

run_mpi_cluster:
# 	for i in $$(seq 1 1); do \
		mpirun -np $(NP) --hostfile $(HOSTFILE) -x OMP_NUM_THREADS=$(THREADS) ./$(PIPE_MPI) $(NODES); \
#  	done
	mpiexec -f $(HOSTFILE) -n $(NP) -env OMP_NUM_THREADS $(THREADS) ./$(PIPE_MPI) $(NODES)

clean:
	rm -f $(SEQ) $(SEQ_OMP) $(PIPE) $(PIPE_MPI)
