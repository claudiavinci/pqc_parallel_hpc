CC=gcc
MPICC=mpicc

CFLAGS=-O2
OMPFLAGS=-fopenmp

# --------- INCLUDE -------------
COMMON_INC=-I common
INC_SEQ=-I ml-kem-768
INC_OMP=-I ml-kem-768-omp

# --------- BINARI ---------------
SEQ=seq_mlkem
SEQ_OMP=seq_mlkem_omp
PIPE_OMP=pipeline_omp_mlkem
PIPE_OMP_MPI=pipeline_omp_mpi_mlkem

# --------- BUILD ----------------
all: seq seq_omp pipe_omp pipe_omp_mpi

seq:
	$(CC) $(CFLAGS) seq_mlkem.c report.c ml-kem-768/*.c common/*.c $(INC_SEQ) $(COMMON_INC) -o $(SEQ)

seq_omp:
	$(CC) $(CFLAGS) $(OMPFLAGS) seq_mlkem.c report.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) -o $(SEQ_OMP)

pipe_omp:
	$(CC) $(CFLAGS) $(OMPFLAGS) pipeline_omp_mlkem.c report.c pipeline.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) -o $(PIPE_OMP)

pipe_omp_mpi:
	$(MPICC) $(CFLAGS) $(OMPFLAGS) pipeline_omp_mlkem.c report.c pipeline.c ml-kem-768-omp/*.c common/*.c $(INC_OMP) $(COMMON_INC) -o $(PIPE_OMP_MPI)

# -------------- RUN SEQUENTIAL ----------

run_seq:
	for i in $$(seq 1 10); do \
		./$(SEQ); \
	done

run_seq_omp:
	for i in $$(seq 1 10); do \
		OMP_NUM_THREADS=$(THREADS) ./$(SEQ_OMP); \
	done

run_pipe_omp:
	for i in $$(seq 1 10); do \
		OMP_NUM_THREADS=$(THREADS) ./$(PIPE_OMP); \
	done

run_pipe_omp_mpi:
	for i in $$(seq 1 10); do \
		mpirun -np $(NP) OMP_NUM_THREADS=$(THREADS) ./$(PIPE_OMP_MPI); \
	done

clean:
	rm -f $(SEQ) $(SEQ_OMP) $(PIPE_OMP) $(PIPE_OMP_MPI)
