# pqc_parallel_hpc

Parallelization of Post-Quantum Cryptography ML-KEM-768 Workload using MPI and OpenMP.

## Requirements

On Ubuntu/Debian:

```bash
sudo apt update

# GCC, G++, Make (includes OpenMP support)
sudo apt install build-essential

# MPI
sudo apt install mpich libmpich-dev

# SSH server
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

## SSH Passwordless Authentication

Generate an SSH key on **each node**:

```bash
ssh-keygen -t ed25519
```

Copy the public key to the other nodes:

```bash
ssh-copy-id username@IP_SERVER
```

Verify that you can log in without entering a password:

```bash
ssh username@IP_SERVER
```

## SSH Configuration (Optional)

To simplify SSH commands, edit:

```bash
nano ~/.ssh/config
```

Example configuration:

```text
Host node1
    HostName 192.168.1.10
    User username

Host node2
    HostName 192.168.1.11
    User username
```

Then you can connect simply with:

```bash
ssh node1
ssh node2
```

This is important to ensure correct username usage for Hydra (MPICH)
Use these hostnames in hosts.txt file when running on MPI cluster

## Build

Compile all executables:

```bash
make all
```
Or compile a specific target:

```bash
make seq
make seq_omp
make pipe
make pipe_mpi
```
## Executables

| Target | Description |
|--------|-------------|
| `seq` | Sequential implementation of ML-KEM |
| `seq_omp` | Sequential implementation with OpenMP parallelization |
| `pipe` | Pipeline implementation using OpenMP |
| `pipe_mpi` | Hybrid MPI + OpenMP pipeline implementation |

## Running

### Sequential

```bash
make run_seq
```

### OpenMP

Specify the number of threads:

```bash
make run_seq_omp THREADS=4
```

or

```bash
make run_pipe THREADS=8
```

### MPI (single machine)

Specify the number of MPI processes and OpenMP threads:

```bash
make run_mpi_local NP=4 THREADS=2
```

### MPI Cluster

Configure the hosts file:

```text
192.168.1.10
192.168.1.11
```
or hostnames if defined in ~/.ssh/config:

```text
node 1
node 2
```

Then run:

```bash
make run_mpi_cluster NP=4 THREADS=2 HOSTFILE=hosts.txt NODES=2
```

## Benchmark Suite

Execute all benchmarks:

```bash
make run_all_tests
```

## Clean

Remove compiled binaries:

```bash
make clean
```