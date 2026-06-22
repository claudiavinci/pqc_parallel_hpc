import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import json
import seaborn as sns

N_JOBS = 100000
REPORT_DIR = f"./report_out/{N_JOBS}_JOBS/"
PLOT_DIR = f"./plots/{N_JOBS}_JOBS/"

def speedup_efficiency(df: DataFrame, baseline, best_res, df_name):
    df["SPEEDUP"] = (baseline / df["TIME_SEC"]).round(3)
    df["EFFICIENCY"] = (df["SPEEDUP"] / df["TOT_WORKERS"]).round(3)

    idx_max_speedup = df["SPEEDUP"].idxmax()
    idx_max_eff = df["EFFICIENCY"].idxmax()

    row_max_speedup = df.loc[idx_max_speedup]
    row_max_eff = df.loc[idx_max_eff]

    best_res[df_name] = {
                "speedup": {
                    "value": row_max_speedup["SPEEDUP"],
                    "time": row_max_speedup["TIME_SEC"],
                    "efficiency": row_max_speedup["EFFICIENCY"],
                    "throughput": row_max_speedup["THROUGHPUT_JS"],
                    "threads": int(row_max_speedup["OMP_THREADS"]),
                },
                "efficiency": {
                    "value": row_max_eff["EFFICIENCY"],
                    "time": row_max_eff["TIME_SEC"],
                    "speedup": row_max_eff["SPEEDUP"],
                    "throughput": row_max_eff["THROUGHPUT_JS"],
                    "threads": int(row_max_eff["OMP_THREADS"]),
                }
            }

    match df_name:
        case "pipe_mpi": 
            best_res[df_name]["speedup"].update({
                "mpi_ranks": int(row_max_speedup["MPI_RANKS"]),
                "tot_workers": int(row_max_speedup["TOT_WORKERS"]),
            })

            best_res[df_name]["efficiency"].update({
                "mpi_ranks": int(row_max_eff["MPI_RANKS"]),
                "tot_workers": int(row_max_eff["TOT_WORKERS"]),
            })

        case "pipe_mpi_cluster": 
            best_res[df_name]["speedup"].update({
                "mpi_ranks": int(row_max_speedup["MPI_RANKS"]),
                "tot_workers": int(row_max_speedup["TOT_WORKERS"]),
                "nodes": int(row_max_speedup["NODES"]),
            })
            best_res[df_name]["efficiency"].update({
                "mpi_ranks": int(row_max_eff["MPI_RANKS"]),
                "tot_workers": int(row_max_eff["TOT_WORKERS"]),
                "nodes": int(row_max_eff["NODES"]),
            })

def plot_metrics(df1, df2, colx, coly, title: str, xlabel: str, ylabel: str, savepath:str):
    plt.figure()
    
    plt.plot(
        df1[colx],
        df1[coly], 
        marker="o",
        label="Sequential (OMP)"
    )

    plt.plot(
        df2[colx],
        df2[coly],
        marker="o", 
        label="Pipeline (OMP)"
    )

    if coly == "SPEEDUP":
        plt.plot(
            df2["TOT_WORKERS"],
            df2["TOT_WORKERS"],
            "--",
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label="Ideal speedup"
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_heatmap(df, title, figname):

    speedup_pivot = df.pivot_table(
        index = "MPI_RANKS",
        columns = "OMP_THREADS",
        values = "SPEEDUP"
    ).sort_index().sort_index(axis=1)

    eff_pivot = df.pivot_table(
        index = "MPI_RANKS",
        columns = "OMP_THREADS",
        values = "EFFICIENCY"
    ).sort_index().sort_index(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        speedup_pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Speedup"},
        annot_kws={"size": 11},
        ax=ax1
    )
    ax1.set_title("Speedup")
    ax1.set_xlabel("OMP Threads")
    ax1.set_ylabel("MPI Ranks")

    sns.heatmap(
    eff_pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Efficiency"},
        annot_kws={"size": 11},
        ax=ax2
    )
    ax2.set_title("Efficiency")
    ax2.set_xlabel("OMP Threads")
    ax2.set_ylabel("MPI Ranks")

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}{figname}")
    plt.close()

if __name__ == "__main__":
    
    # -------------- REPORTS READING ------------------------
    print(f"Reading results from {REPORT_DIR}...")

    seq_df = pd.read_csv(f"{REPORT_DIR}seq_mlkem_results.csv")
    baseline = seq_df[seq_df["OMP_ENABLED"] == 0]["TIME_SEC"].mean().round(3)
    seq_omp_df = seq_df[seq_df["OMP_ENABLED"] == 1].copy().reset_index(drop=True)

    pipe_df = pd.read_csv(f"{REPORT_DIR}pipeline_results.csv")
    
    pipe_mpi_df = pd.read_csv(f"{REPORT_DIR}pipeline_mpi_results.csv")
    pipe_mpi_cluster_df = pipe_mpi_df[pipe_mpi_df["NODES"] > 1].copy().reset_index(drop=True)
    pipe_mpi_df = pipe_mpi_df[pipe_mpi_df["NODES"] == 1].reset_index(drop=True)

    # -------------- MEAN VALUES FOR EACH CONFIG ------------------------
    
    seq_omp_mean = seq_omp_df.groupby(["TOT_WORKERS", "OMP_THREADS"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean"), THROUGHPUT_JS=("THROUGHPUT_JS", "mean")).round(3).sort_values("OMP_THREADS").reset_index(drop=True)
    pipe_mean = pipe_df.groupby(["TOT_WORKERS", "OMP_THREADS"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean"), THROUGHPUT_JS=("THROUGHPUT_JS", "mean")).round(3).sort_values("OMP_THREADS").reset_index(drop=True)
    pipe_mpi_mean = pipe_mpi_df.groupby(["TOT_WORKERS", "MPI_RANKS", "OMP_THREADS"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean"), THROUGHPUT_JS=("THROUGHPUT_JS", "mean")).round(3).sort_values("TOT_WORKERS").reset_index(drop=True)
    pipe_mpi_cluster_mean = pipe_mpi_cluster_df.groupby(["TOT_WORKERS", "MPI_RANKS", "OMP_THREADS", "NODES"], as_index=False).agg(TIME_SEC=("TIME_SEC", "mean"), THROUGHPUT_JS=("THROUGHPUT_JS", "mean")).round(3).sort_values("TOT_WORKERS").reset_index(drop=True)
    
    # # -------------- SPEEDUP, EFFICIENCY AND BEST RESULTS ------------------------
    
    best_res = {
        "seq_omp": None,
        "pipe": None,
        "pipe_mpi": None,
        "pipe_mpi_cluster": None,
    }

    speedup_efficiency(seq_omp_mean, baseline, best_res, "seq_omp")
    speedup_efficiency(pipe_mean, baseline, best_res, "pipe")
    speedup_efficiency(pipe_mpi_mean, baseline, best_res, "pipe_mpi")
    speedup_efficiency(pipe_mpi_cluster_mean, baseline, best_res, "pipe_mpi_cluster")

    # print(json.dumps(best_res, indent=4, ensure_ascii=False))

    with open(f"{PLOT_DIR}best_results.json", "w") as f: 
        json.dump(best_res, f, indent=4)
    print(f"Saved JSON to {PLOT_DIR}")

    # print(seq_omp_mean.head())
    # print(pipe_mean.head())
    # print(pipe_mpi_mean.head())
    # print(pipe_mpi_cluster_mean.head())
    
    # -------------- METRICS PLOTS ------------------------

    plot_metrics(seq_omp_mean, 
                 pipe_mean,  
                 colx="TOT_WORKERS", 
                 coly="TIME_SEC", 
                 title="Execution Time Scaling", 
                 xlabel="Total Workers", 
                 ylabel="Time (s)", 
                 savepath=f"{PLOT_DIR}exec_time_scaling.png")
    
    plot_metrics(seq_omp_mean, 
                 pipe_mean, 
                 colx="TOT_WORKERS", 
                 coly="SPEEDUP", 
                 title="Parallel Speedup", 
                 xlabel="Total Workers", 
                 ylabel="Speedup", 
                 savepath=f"{PLOT_DIR}speedup.png")
    
    plot_metrics(seq_omp_mean, 
                 pipe_mean, 
                 colx="TOT_WORKERS", 
                 coly="EFFICIENCY", 
                 title="Parallel Efficiency", 
                 xlabel="Total Workers", 
                 ylabel="Efficiency", 
                 savepath=f"{PLOT_DIR}efficiency.png")
    
    plot_metrics(seq_omp_mean, 
                 pipe_mean, 
                 colx="TOT_WORKERS", 
                 coly="THROUGHPUT_JS", 
                 title="Throughput Scaling", 
                 xlabel="Total Workers", 
                 ylabel="Throughput (job/s)", 
                 savepath=f"{PLOT_DIR}throughput_scaling.png")
    
    plot_metrics(seq_omp_mean, 
                pipe_mean, 
                colx="TOT_WORKERS", 
                coly="THROUGHPUT_JS", 
                title="Throughput Scaling", 
                xlabel="Total Workers", 
                ylabel="Throughput (job/s)", 
                savepath=f"{PLOT_DIR}throughput_scaling.png")
    
    # # -------------- MPI + OPENMP HEATMAP ------------------------
    
    # plot_heatmap(pipe_mpi_mean, "SPEEDUP", "MPI+OMP Speedup Heatmap (1 node)" ,"heatmap_speedup_local.png")
    # plot_heatmap(pipe_mpi_mean, "EFFICIENCY", "MPI+OMP Efficiency Heatmap (1 node)" ,"heatmap_eff_local.png")
    # plot_heatmap(pipe_mpi_cluster_mean, "SPEEDUP", "MPI+OMP Speedup Heatmap (2 nodes)" ,"heatmap_speedup_cluster.png")
    # plot_heatmap(pipe_mpi_cluster_mean, "EFFICIENCY", "MPI+OMP Efficiency Heatmap (2 nodes)" ,"heatmap_eff_cluster.png")

    plot_heatmap(pipe_mpi_mean, "MPI+OMP Performance Heatmaps (1 node)" ,"heatmaps_local.png")
    plot_heatmap(pipe_mpi_cluster_mean, "MPI+OMP Performance Heatmaps (2 nodes)" ,"heatmaps_cluster.png")

    # # pipe_mpi_fixed = pipe_mpi_mean[pipe_mpi_mean["OMP_THREADS"] == 3].copy()



    # # 3. Plot Pipeline Hybrid (MPI + OMP) diviso per configurazione
    # # Usiamo seaborn per tracciare una linea diversa per ogni numero di MPI_RANKS
    # sns.lineplot(
    #     data=pip_omp_mpi,
    #     x=colx,
    #     y=coly,
    #     hue="MPI_RANKS",       # Crea una curva diversa per ogni numero di rank
    #     palette="viridis",     # Scala di colori coerente
    #     marker="D",
    #     legend=True
    # )
