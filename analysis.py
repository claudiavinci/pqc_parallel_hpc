import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pandas import DataFrame 
import numpy as np
import json
import seaborn as sns

N_JOBS = 100000
# REPORT_DIR = f"./report_out/{N_JOBS}_JOBS/"
# ANALYSIS_DIR = f"./analysis/{N_JOBS}_JOBS/"

REPORT_DIR = f"./report_out/flto_O3/no_bind/"
ANALYSIS_DIR = f"./analysis/flto_O3/no_bind/"

PLOTS_DIR = f"{ANALYSIS_DIR}plots/"
METRICS_DIR = f"{ANALYSIS_DIR}metrics/"

TIME_METRICS = {
    "TOTAL": "TIME_SEC",
    "KEYGEN": "KEYGEN_SEC",
    "ENC": "ENC_SEC",
    "DEC": "DEC_SEC",
}

def preprocess_dataframe(df:DataFrame, groupby_cols:list):
    return df.groupby(groupby_cols, as_index=False).agg(TIME_SEC=("TIME_SEC", "median"), THROUGHPUT_JS=("THROUGHPUT_JS", "median")).round(3).sort_values("TOT_WORKERS").reset_index(drop=True)

def add_speedup_efficiency(df: DataFrame, baseline, metric):
    speedup = f"{metric}_SPEEDUP"
    efficiency = f"{metric}_EFFICIENCY"

    df[speedup] = (baseline / df[metric]).round(3)

    df[efficiency] = (df[speedup] / df["TOT_WORKERS"]).round(3)

def extract_results(row, metric, kind):
    speedup = f"{metric}_SPEEDUP"
    efficiency = f"{metric}_EFFICIENCY"

    if kind == "speedup":
        return {
            "value": row[speedup],
            "time": row[metric],
            "efficiency": row[efficiency],
            "throughput": row["THROUGHPUT_JS"],
            "threads": int(row["OMP_THREADS"]),
        }

    elif kind == "efficiency":
        return {
            "value": row[efficiency],
            "time": row[metric],
            "speedup": row[speedup],
            "throughput": row["THROUGHPUT_JS"],
            "threads": int(row["OMP_THREADS"]),
        }

    else:
        return {
            "value": row["THROUGHPUT_JS"],
            "time": row[metric],
            "speedup": row[speedup],
            "efficiency": row[efficiency],
            "threads": int(row["OMP_THREADS"]),
        }

def add_mpi_info(result, row, df_name):
    result["tot_workers"] = int(row["TOT_WORKERS"])
    if "mpi" in df_name:
        result["mpi_ranks"] = int(row["MPI_RANKS"])
    if "cluster" in df_name:
        result["nodes"] = int(row["NODES"])

def evaluate_metrics(df: DataFrame, baseline, df_name):
    results = {}
    for name, metric in TIME_METRICS.items():
        add_speedup_efficiency(df, baseline, metric)
        best_speedup = df.loc[df[f"{metric}_SPEEDUP"].idxmax()]
        best_efficiency = df.loc[df[f"{metric}_EFFICIENCY"].idxmax()]
        best_throughput = df.loc[df["THROUGHPUT_JS"].idxmax()]

        results[name] = {
            "speedup": extract_results(best_speedup, metric, "speedup"),
            "efficiency": extract_results(best_efficiency, metric, "efficiency"),
            "throughput": extract_results(best_throughput, metric, "throughput"),
        }

        add_mpi_info(results[name]["speedup"], best_speedup, df_name)
        add_mpi_info(results[name]["efficiency"], best_efficiency, df_name)
        add_mpi_info(results[name]["throughput"], best_throughput, df_name)

    return results

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

    if "SPEEDUP" in coly:
        plt.plot(
            df2["TOT_WORKERS"],
            df2["TOT_WORKERS"],
            "--",
            marker="o", 
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label="Ideal speedup"
        )
    plt.xticks(sorted(df1["TOT_WORKERS"].unique()))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_heatmap(df, metric, title, savepath):
    pivot = df.pivot_table(
        index = "MPI_RANKS",
        columns = "OMP_THREADS",
        values = metric
    )

    plt.figure()
    current_cmap = plt.get_cmap("viridis").copy()
    current_cmap.set_bad(color="lightgray")
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=current_cmap,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": metric.capitalize()},
        annot_kws={"size": 12},
    )

    max_row_idx, max_col_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.shape)
    
    # 2. Crea un rettangolo da sovrapporre alla cella
    # Nota: in Matplotlib/Seaborn le colonne sono sull'asse X e le righe sull'asse Y
    rect = Rectangle(
        (max_col_idx, max_row_idx), # Punto di partenza (angolo in alto a sinistra della cella)
        1, 1,                       # Larghezza e altezza del rettangolo (1 sola cella)
        fill=False,                 # Non riempire l'interno, vogliamo solo il bordo
        edgecolor="red",            # Colore del bordo (puoi usare anche "magenta", "white", ecc.)
        linewidth=3,                # Spessore del bordo
        linestyle="-"               # Stile della linea
    )
    
    # 3. Aggiunge il rettangolo all'asse grafico
    ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlabel("OMP Threads")
    ax.set_ylabel("MPI Ranks")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_iso_mpi(df, coly, title, savepath):

    plt.figure()

    for rank in sorted(df["MPI_RANKS"].unique()):
        sub = df[df["MPI_RANKS"] == rank].sort_values("OMP_THREADS")

        plt.plot(
            sub["OMP_THREADS"],
            sub[coly],
            marker="o",
            label=f"MPI Ranks={rank}"
        )

    plt.xticks(sorted(df["OMP_THREADS"].unique()))
    plt.title(title)
    plt.xlabel("OMP Threads")
    plt.ylabel(coly)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

if __name__ == "__main__":
    
    # -------------- REPORTS READING ------------------------
    print(f"Reading results from {REPORT_DIR}...")

    seq_df = pd.read_csv(f"{REPORT_DIR}/seq_mlkem_results.csv")
    baseline = seq_df[seq_df["OMP_ENABLED"] == 0]["TIME_SEC"].mean().round(3)
    seq_omp_df = seq_df[seq_df["OMP_ENABLED"] == 1].copy().reset_index(drop=True)

    pipe_df = pd.read_csv(f"{REPORT_DIR}/pipeline_results.csv")
    
    pipe_mpi_df = pd.read_csv(f"{REPORT_DIR}/pipeline_mpi_results.csv")
    pipe_mpi_cluster_df = pipe_mpi_df[pipe_mpi_df["NODES"] > 1].copy().reset_index(drop=True)
    pipe_mpi_df = pipe_mpi_df[pipe_mpi_df["NODES"] == 1].reset_index(drop=True)

    # -------------- MEAN VALUES FOR EACH CONFIG ------------------------
    
    seq_omp_mean = preprocess_dataframe(seq_omp_df, ["TOT_WORKERS", "OMP_THREADS"])
    pipe_mean = preprocess_dataframe(pipe_df, ["TOT_WORKERS", "OMP_THREADS"])
    pipe_mpi_mean = preprocess_dataframe(pipe_mpi_df, ["TOT_WORKERS", "MPI_RANKS", "OMP_THREADS"])
    pipe_mpi_cluster_mean = preprocess_dataframe(pipe_mpi_cluster_df, ["TOT_WORKERS", "MPI_RANKS", "OMP_THREADS", "NODES"])
    
    # # -------------- SPEEDUP, EFFICIENCY AND BEST RESULTS ------------------------
    
    best_res = {
        "seq_omp": evaluate_metrics(seq_omp_mean, baseline, "seq_omp"),
        "pipe": evaluate_metrics(pipe_mean, baseline, "pipe"),
        "pipe_mpi": evaluate_metrics(pipe_mpi_mean, baseline, "pipe_mpi"),
        "pipe_mpi_cluster": evaluate_metrics(pipe_mpi_cluster_mean, baseline, "pipe_mpi_cluster"),
    }

    with open(f"{ANALYSIS_DIR}/best_results.json", "w") as f: 
        json.dump(best_res, f, indent=4)
    print(f"Saved JSON to {ANALYSIS_DIR}")

    seq_omp_mean.to_csv(f"{METRICS_DIR}seq_omp_metrics.csv", index=False)
    pipe_mean.to_csv(f"{METRICS_DIR}pipe_metrics.csv", index=False)
    pipe_mpi_mean.to_csv(f"{METRICS_DIR}pipe_mpi_metrics.csv", index=False)
    pipe_mpi_cluster_mean.to_csv(f"{METRICS_DIR}pipe_mpi_cluster_metrics.csv", index=False)

    # -------------- COMPLETE WORKFLOW EVALUATION -----------------------------------------
    
    # -------------- METRICS PLOTS ------------------------
    
    # plot_metrics(seq_omp_mean, 
    #              pipe_mean, 
    #              colx="TOT_WORKERS", 
    #              coly="SPEEDUP", 
    #              title="Sequential OMP vs. Pipeline OMP Speedup", 
    #              xlabel="Total Workers (Threads)", 
    #              ylabel="Speedup", 
    #              savepath=f"{PLOTS_DIR}speedup.png")
    
    # plot_metrics(seq_omp_mean, 
    #              pipe_mean, 
    #              colx="TOT_WORKERS", 
    #              coly="EFFICIENCY", 
    #              title="Sequential OMP vs. Pipeline OMP Efficiency", 
    #              xlabel="Total Workers (Threads)", 
    #              ylabel="Efficiency", 
    #              savepath=f"{PLOTS_DIR}efficiency.png")
    
    # plot_metrics(seq_omp_mean, 
    #              pipe_mean, 
    #              colx="TOT_WORKERS", 
    #              coly="THROUGHPUT_JS", 
    #              title="Sequential OMP vs. Pipeline OMP Throughput", 
    #              xlabel="Total Workers (Threads)", 
    #              ylabel="Throughput (job/s)", 
    #              savepath=f"{PLOTS_DIR}throughput_scaling.png")

    # # # -------------- MPI + OPENMP HEATMAP ------------------------

    # plot_heatmap(pipe_mpi_mean, "SPEEDUP", "MPI+OMP Speedup (1 node)", f"{PLOTS_DIR}/mpi_local/heatmap_speedup.png")
    # plot_heatmap(pipe_mpi_mean, "EFFICIENCY", "MPI+OMP Efficiency (1 node)", f"{PLOTS_DIR}/mpi_local/heatmap_eff.png")
    # plot_heatmap(pipe_mpi_mean, "THROUGHPUT_JS", "MPI+OMP Throughput (1 node)", f"{PLOTS_DIR}/mpi_local/heatmap_through.png")

    # plot_iso_mpi(pipe_mpi_mean, "SPEEDUP", "OMP+MPI Speedup (1 node)", f"{PLOTS_DIR}/mpi_local/speedup.png")
    # plot_iso_mpi(pipe_mpi_mean, "EFFICIENCY", "OMP+MPI Efficiency (1 node)", f"{PLOTS_DIR}/mpi_local/efficiency.png")
    # plot_iso_mpi(pipe_mpi_mean, "THROUGHPUT_JS", "OMP+MPI Throughput (1 node)", f"{PLOTS_DIR}/mpi_local/throughput.png")

    # plot_heatmap(pipe_mpi_cluster_mean, "SPEEDUP", "MPI+OMP Speedup (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/heatmap_speedup.png")
    # plot_heatmap(pipe_mpi_cluster_mean, "EFFICIENCY", "MPI+OMP Efficiency (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/heatmap_eff.png")
    # plot_heatmap(pipe_mpi_cluster_mean, "THROUGHPUT_JS", "MPI+OMP Throughput (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/heatmap_through.png")

    # plot_iso_mpi(pipe_mpi_cluster_mean, "SPEEDUP", "OMP+MPI Speedup (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/speedup.png")
    # plot_iso_mpi(pipe_mpi_cluster_mean, "EFFICIENCY", "OMP+MPI Efficiency (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/efficiency.png")
    # plot_iso_mpi(pipe_mpi_cluster_mean, "THROUGHPUT_JS", "OMP+MPI Throughput (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/throughput.png")

# ---------------------- STAGES EVALUATION ---------------------------- 

    # plot_metrics(seq_omp_mean, 
    #              pipe_mean, 
    #              colx="TOT_WORKERS", 
    #              coly="KEYGEN_SEC_SPEEDUP", 
    #              title="Sequential OMP vs. Pipeline OMP Keygen Speedup", 
    #              xlabel="Total Workers (Threads)", 
    #              ylabel="Speedup", 
    #              savepath=f"{PLOTS_DIR}keygen_speedup.png")

    # plot_metrics(seq_omp_mean, 
    #             pipe_mean, 
    #             colx="TOT_WORKERS", 
    #             coly="ENC_SEC_SPEEDUP", 
    #             title="Sequential OMP vs. Pipeline OMP Encapsulation Speedup", 
    #             xlabel="Total Workers (Threads)", 
    #             ylabel="Speedup", 
    #             savepath=f"{PLOTS_DIR}enc_speedup.png")
    
    # plot_metrics(seq_omp_mean, 
    #             pipe_mean, 
    #             colx="TOT_WORKERS", 
    #             coly="DEC_SEC_SPEEDUP", 
    #             title="Sequential OMP vs. Pipeline OMP Decapsulation Speedup", 
    #             xlabel="Total Workers (Threads)", 
    #             ylabel="Speedup", 
    #             savepath=f"{PLOTS_DIR}dec_speedup.png")
    
    # plot_metrics(seq_omp_mean, 
    #              pipe_mean, 
    #              colx="TOT_WORKERS", 
    #              coly="KEYGEN_SEC_EFFICIENCY", 
    #              title="Sequential OMP vs. Pipeline OMP Keygen Efficiency", 
    #              xlabel="Total Workers (Threads)", 
    #              ylabel="Efficiency", 
    #              savepath=f"{PLOTS_DIR}keygen_efficiency.png")

    # plot_metrics(seq_omp_mean, 
    #             pipe_mean, 
    #             colx="TOT_WORKERS", 
    #             coly="ENC_SEC_EFFICIENCY", 
    #             title="Sequential OMP vs. Pipeline OMP Encapsulation Efficiency", 
    #             xlabel="Total Workers (Threads)", 
    #             ylabel="Efficiency", 
    #             savepath=f"{PLOTS_DIR}enc_efficiency.png")
    
    # plot_metrics(seq_omp_mean, 
    #             pipe_mean, 
    #             colx="TOT_WORKERS", 
    #             coly="DEC_SEC_EFFICIENCY", 
    #             title="Sequential OMP vs. Pipeline OMP Decapsulation Efficiency", 
    #             xlabel="Total Workers (Threads)", 
    #             ylabel="Efficiency", 
    #             savepath=f"{PLOTS_DIR}dec_efficiency.png")