import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import json
import seaborn as sns

REPORT_DIR = "./reports/"
PLOT_DIR = "./plots/"

def speedup_efficiency(df: DataFrame, baseline, best_res, df_name):
    df["SPEEDUP"] = baseline / df["TIME_SEC"]
    df["EFFICIENCY"] = df["SPEEDUP"] / df["TOT_WORKERS"]

    idx_max_speedup = df["SPEEDUP"].idxmax()
    idx_max_eff = df["EFFICIENCY"].idxmax()

    row_max_speedup = df.loc[idx_max_speedup]
    row_max_eff = df.loc[idx_max_eff]

    best_res[df_name] = {
        "speedup": {
            "value": float(row_max_speedup["SPEEDUP"]),
            "time_s": float(row_max_speedup["TIME_SEC"]),
            "nodes": float(row_max_speedup["MPI_RANKS"]),
            "threads": float(row_max_speedup["OMP_THREADS"]),
            "tot_workers": float(row_max_speedup["TOT_WORKERS"]),
        },
        "efficiency": {
            "value": float(row_max_eff["EFFICIENCY"]),
            "time_s": float(row_max_eff["TIME_SEC"]),
            "nodes": float(row_max_eff["MPI_RANKS"]),
            "threads": float(row_max_eff["OMP_THREADS"]),
            "tot_workers": float(row_max_eff["TOT_WORKERS"]),
        }
    }

def plot_metrics(seq_omp, pip_omp, pip_omp_mpi, colx, coly, title: str, xlabel: str, ylabel: str, savepath:str):
    plt.figure()
    
    plt.plot(
        seq_omp[colx],
        seq_omp[coly], 
        label="Sequential (OMP)"
    )

    plt.plot(
        pip_omp[colx],
        pip_omp[coly], 
        label="Pipeline (OMP)"
    )

    # 3. Plot Pipeline Hybrid (MPI + OMP) diviso per configurazione
    # Usiamo seaborn per tracciare una linea diversa per ogni numero di MPI_RANKS
    sns.lineplot(
        data=pip_omp_mpi,
        x=colx,
        y=coly,
        hue="MPI_RANKS",       # Crea una curva diversa per ogni numero di rank
        palette="viridis",     # Scala di colori coerente
        marker="D",
        legend=True
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    # plt.legend()

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

if __name__ == "__main__":

    # -------------- REPORTS READING ------------------------

    seq_df = pd.read_csv(f"{REPORT_DIR}seq_mlkem_results.csv")
    pipe_omp_df = pd.read_csv(f"{REPORT_DIR}pipeline_omp_results.csv")
    pipe_mpi_omp_df = pd.read_csv(f"{REPORT_DIR}pipeline_omp_mpi_results.csv")
    
    baseline = seq_df[seq_df["OMP_ENABLED"] == 0]["TIME_SEC"].mean()
    seq_omp_df = seq_df[seq_df["OMP_ENABLED"] == 1].copy() 

    # -------------- MEAN VALUES FOR EACH CONFIG ------------------------
    
    seq_omp_mean = seq_omp_df.groupby(["TOT_WORKERS"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean")).sort_values("TOT_WORKERS").reset_index(drop=True)
    pipe_omp_mean = pipe_omp_df.groupby(["TOT_WORKERS"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean")).sort_values("TOT_WORKERS").reset_index(drop=True)
    pipe_mpi_omp_mean = pipe_mpi_omp_df.groupby(["TOT_WORKERS", "MPI_RANKS", "OMP_THREADS"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean")).sort_values("TOT_WORKERS").reset_index(drop=True)
    
    # -------------- SPEEDUP, EFFICIENCY AND BEST RESULTS ------------------------
    
    best_res = {
        "seq_omp": None,
        "pipe_omp": None,
        "pipe_mpi_omp": None
    }

    speedup_efficiency(seq_omp_mean, baseline, best_res, "seq_omp")
    speedup_efficiency(pipe_omp_mean, baseline, best_res, "pipe_omp")
    speedup_efficiency(pipe_mpi_omp_mean, baseline, best_res, "pipe_mpi_omp")

    with open(f"{PLOT_DIR}best_results.json", "w") as f: 
        json.dump(best_res, f, indent=4)
    print(f"Saved JSON to {PLOT_DIR}")
    
    # -------------- MPI + OPENMP HEATMAP ------------------------
    
    pivot = pipe_mpi_omp_mean.pivot_table(
        index = "MPI_RANKS",
        columns = "OMP_THREADS",
        values = "TIME_SEC"
    )
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure()
    sns.heatmap(pivot, annot=True, fmt=".2f")
    plt.title("MPI + OMP Time Heatmap")
    plt.xlabel("OMP Threads")
    plt.ylabel("MPI Ranks")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}heatmap_mpi_omp.png")
    plt.close()
    
    # -------------- METRICS PLOTS ------------------------
    pipe_mpi_omp_fixed = pipe_mpi_omp_mean[pipe_mpi_omp_mean["OMP_THREADS"] == 3].copy()

    plot_metrics(seq_omp_mean, 
                 pipe_omp_mean, 
                 pipe_mpi_omp_fixed, 
                 colx="TOT_WORKERS", 
                 coly="TIME_SEC", 
                 title="Execution Time Scaling", 
                 xlabel="Total Workers", 
                 ylabel="Time (s)", 
                 savepath=f"{PLOT_DIR}exec_time_scaling.png")
    
    plot_metrics(seq_omp_mean, 
                 pipe_omp_mean, 
                 pipe_mpi_omp_fixed, 
                 colx="TOT_WORKERS", 
                 coly="SPEEDUP", 
                 title="Parallel Speedup", 
                 xlabel="Total Workers", 
                 ylabel="Speedup", 
                 savepath=f"{PLOT_DIR}speedup.png")
    
    plot_metrics(seq_omp_mean, 
                 pipe_omp_mean, 
                 pipe_mpi_omp_fixed, 
                 colx="TOT_WORKERS", 
                 coly="EFFICIENCY", 
                 title="Parallel Efficiency", 
                 xlabel="Total Workers", 
                 ylabel="Efficiency", 
                 savepath=f"{PLOT_DIR}efficiency.png")
    
    plot_metrics(seq_omp_mean, 
                 pipe_omp_mean, 
                 pipe_mpi_omp_fixed, 
                 colx="TOT_WORKERS", 
                 coly="TROUGHPUT_JS", 
                 title="Throughput Scaling", 
                 xlabel="Total Workers", 
                 ylabel="Troughput (job/s)", 
                 savepath=f"{PLOT_DIR}troughput_scaling.png")
    


