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

REPORT_DIR = f"./report_out/flto_O3/"
ANALYSIS_DIR = f"./analysis/flto_O3/"

PLOTS_DIR = f"{ANALYSIS_DIR}plots/"
METRICS_DIR = f"{ANALYSIS_DIR}metrics/"

TIME_METRICS = {
    "TOTAL": "TIME_SEC",
    "KEYGEN": "KEYGEN_SEC",
    "ENC": "ENC_SEC",
    "DEC": "DEC_SEC",
}

def compute_baseline(df: DataFrame):
    baselines = {}
    for name, metric in TIME_METRICS.items():
        baseline = df[df["OMP_ENABLED"] == 0][metric].mean().round(3)
        baselines[name] = baseline
    return baselines

def baseline_metrics(df: DataFrame, baselines: dict):
    # Calcola il rapporto tra il tempo del collo di bottiglia e lo stadio più veloce
    keygen_st_mean = (baselines["KEYGEN"]/ N_JOBS) * 1000  # Converti in millisecondi
    enc_st_mean = (baselines["ENC"]/ N_JOBS) * 1000  # Converti in millisecondi
    dec_st_mean = (baselines["DEC"]/ N_JOBS) * 1000  # Converti in millisecondi

    max_stage_time = max(keygen_st_mean, enc_st_mean, dec_st_mean)
    min_stage_time = min(keygen_st_mean, enc_st_mean, dec_st_mean)
    balance_ratio = (max_stage_time / min_stage_time).round(3)

    throughput_baseline = (N_JOBS / baselines["TOTAL"]).round(3)
    throughput_per_stage = {
        "KEYGEN": (N_JOBS / baselines["KEYGEN"]).round(3),
        "ENC": (N_JOBS / baselines["ENC"]).round(3),
        "DEC": (N_JOBS / baselines["DEC"]).round(3),
    }
    
    baseline_metrics = {
        "service_times": {
            "KEYGEN": keygen_st_mean,
            "ENC": enc_st_mean,
            "DEC": dec_st_mean
        },
        "balance_ratio": balance_ratio,
        "throughput_baseline": throughput_baseline,
        "throughput_per_stage": throughput_per_stage
    }
    with open(f"{METRICS_DIR}/baseline_metrics.json", "w") as f: 
        json.dump(baseline_metrics, f, indent=4)
    print(f"Saved JSON to {METRICS_DIR}")
    return df

def preprocess_dataframe(df:DataFrame, groupby_cols:list):
    return df.groupby(groupby_cols, as_index=False).agg(TIME_SEC=("TIME_SEC", "median"), THROUGHPUT_JS=("THROUGHPUT_JS", "median"), KEYGEN_SEC=("KEYGEN_SEC", "median"), ENC_SEC=("ENC_SEC", "median"), DEC_SEC=("DEC_SEC", "median")).round(6).sort_values("TOT_WORKERS").reset_index(drop=True)

def add_speedup_efficiency(df: DataFrame, baseline, metric, name):
    speedup = f"{name}_SPEEDUP"
    efficiency = f"{name}_EFFICIENCY"

    df[speedup] = (baseline / df[metric]).round(3)

    df[efficiency] = (df[speedup] / df["TOT_WORKERS"]).round(3)

def extract_results(row, metric, name, kind):
    speedup = f"{name}_SPEEDUP"
    efficiency = f"{name}_EFFICIENCY"

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
    if "mpi" in df_name:
        result["mpi_ranks"] = int(row["MPI_RANKS"])
        result["tot_workers"] = int(row["TOT_WORKERS"])

    if "cluster" in df_name:
        result["nodes"] = int(row["NODES"])
        result["tot_workers"] = int(row["TOT_WORKERS"])

def evaluate_metrics(df: DataFrame, baselines, df_name):
    results = {}
    for name, metric in TIME_METRICS.items():
        add_speedup_efficiency(df, baselines[name], metric, name)
        best_speedup = df.loc[df[f"{name}_SPEEDUP"].idxmax()]
        best_efficiency = df.loc[df[f"{name}_EFFICIENCY"].idxmax()]

        results[name] = {
            "speedup": extract_results(best_speedup, metric, name, "speedup"),
            "efficiency": extract_results(best_efficiency, metric, name, "efficiency"),
        }

        add_mpi_info(results[name]["speedup"], best_speedup, df_name)
        add_mpi_info(results[name]["efficiency"], best_efficiency, df_name)

        if name == "TOTAL":
            best_throughput = df.loc[df["THROUGHPUT_JS"].idxmax()]
            results[name]["throughput"] = extract_results(best_throughput, metric, name, "throughput")
            add_mpi_info(results[name]["throughput"], best_throughput, df_name)

    return results

def plot_metrics(df1, colx, coly, title: str, xlabel: str, ylabel: str, savepath:str, df2 = None):
    plt.figure()
    
    plt.plot(
        df1[colx],
        df1[coly], 
        marker="o",
        label="Sequential (OMP)"
    )

    if df2 is not None:
        plt.plot(
            df2[colx],
            df2[coly],
            marker="o", 
            label="Pipeline (OMP)"
        )

    if "SPEEDUP" in coly:
        plt.plot(
            df1["TOT_WORKERS"],
            df1["TOT_WORKERS"],
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

def plot_omp_metrics(seq_omp_mean, pipe_mean):
    for name, metric in TIME_METRICS.items():
        if name == "TOTAL":
            plot_metrics(seq_omp_mean, 
                        colx="TOT_WORKERS", 
                        coly=f"{name}_SPEEDUP", 
                        title=f"Sequential OMP vs. Pipeline OMP {name.capitalize()} Speedup", 
                        xlabel="Total Workers (Threads)", 
                        ylabel="Speedup", 
                        savepath=f"{PLOTS_DIR}{name.lower()}_speedup.png",
                        df2=pipe_mean
            )
            
            plot_metrics(seq_omp_mean, 
                        colx="TOT_WORKERS", 
                        coly=f"{name}_EFFICIENCY", 
                        title=f"Sequential OMP vs. Pipeline OMP {name.capitalize()} Efficiency", 
                        xlabel="Total Workers (Threads)", 
                        ylabel="Efficiency", 
                        savepath=f"{PLOTS_DIR}{name.lower()}_efficiency.png",
                        df2=pipe_mean
            )

            plot_metrics(seq_omp_mean, 
                colx="TOT_WORKERS", 
                coly="THROUGHPUT_JS", 
                title="Sequential OMP vs. Pipeline OMP Throughput", 
                xlabel="Total Workers (Threads)", 
                ylabel="Throughput (job/s)", 
                savepath=f"{PLOTS_DIR}throughput_scaling.png", 
                df2=pipe_mean, 
            )
        else:
            plot_metrics(seq_omp_mean,
                        colx="TOT_WORKERS",
                        coly=f"{name}_SPEEDUP",
                        title=f"Sequential OMP {name.capitalize()} Speedup",
                        xlabel="Total Workers (Threads)",
                        ylabel="Speedup",
                        savepath=f"{PLOTS_DIR}{name.lower()}_seqomp_speedup.png"
            )

            plot_metrics(seq_omp_mean,
                    colx="TOT_WORKERS",
                    coly=f"{name}_EFFICIENCY",
                    title=f"Sequential OMP {name.capitalize()} Efficiency",
                    xlabel="Total Workers (Threads)",
                    ylabel="Efficiency",
                    savepath=f"{PLOTS_DIR}{name.lower()}_seqomp_efficiency.png"
            )

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

def plot_mpi_metrics(df, label, output_dir):
    metrics = {
        "TOTAL_SPEEDUP": "Speedup",
        "TOTAL_EFFICIENCY": "Efficiency",
        "THROUGHPUT_JS": "Throughput"
    }
    for metric, metric_name in metrics.items():
        plot_heatmap(df, metric, f"MPI+OMP {metric_name} ({label})", f"{output_dir}/heatmap_{metric_name.lower()}.png")
        plot_iso_mpi(df, metric, f"MPI+OMP {metric_name} ({label})", f"{output_dir}/{metric_name.lower()}.png")

def plot_service_times(df: DataFrame, title: str, savepath: str):
    # Bar plot total stage time per ogni configurazione.
    df["KEYGEN_SERVICE_MS"] = df["KEYGEN_SEC"] * 1000 / N_JOBS
    df["ENC_SERVICE_MS"] = df["ENC_SEC"] * 1000 / N_JOBS
    df["DEC_SERVICE_MS"] = df["DEC_SEC"] * 1000 / N_JOBS

    # Ordinamento configurazioni
    if "MPI_RANKS" in df.columns:
        df = df.sort_values(
            ["TOT_WORKERS", "MPI_RANKS", "OMP_THREADS"]
        )
        labels = [f"{r}×{t}\n({r*t})" for r, t in zip(df["MPI_RANKS"], df["OMP_THREADS"])]
        xlabel = "MPI Ranks × OMP Threads (Total Workers)"

    else:
        labels = [str(w) for w in df["TOT_WORKERS"]]
        xlabel = "Total Workers"

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(max(8, len(df)*0.8), 5), dpi=300)

    plt.bar(x - width, df["KEYGEN_SERVICE_MS"], width, label=r"Keygen ($S_{keygen}$)")

    plt.bar(x, df["ENC_SERVICE_MS"], width, label=r"Encapsulation ($S_{enc}$)")

    plt.bar(x + width, df["DEC_SERVICE_MS"], width, label=r"Decapsulation ($S_{dec}$)")

    plt.xticks(x, labels)

    plt.xlabel(xlabel)
    plt.ylabel("Stage Service time (ms)")

    plt.title(title)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


if __name__ == "__main__":
    
    # -------------- REPORTS READING ------------------------
    print(f"Reading results from {REPORT_DIR}...")

    seq_df = pd.read_csv(f"{REPORT_DIR}/seq_mlkem_results.csv")
    baselines = compute_baseline(seq_df)
    baseline_metrics(seq_df, baselines)

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
        "seq_omp": evaluate_metrics(seq_omp_mean, baselines, "seq_omp"),
        "pipe": evaluate_metrics(pipe_mean, baselines, "pipe"),
        "pipe_mpi": evaluate_metrics(pipe_mpi_mean, baselines, "pipe_mpi"),
        "pipe_mpi_cluster": evaluate_metrics(pipe_mpi_cluster_mean, baselines, "pipe_mpi_cluster"),
    }

    with open(f"{ANALYSIS_DIR}/best_results.json", "w") as f: 
        json.dump(best_res, f, indent=4)
    print(f"Saved JSON to {ANALYSIS_DIR}")

    seq_omp_mean.to_csv(f"{METRICS_DIR}seq_omp_metrics.csv", index=False)
    pipe_mean.to_csv(f"{METRICS_DIR}pipe_metrics.csv", index=False)
    pipe_mpi_mean.to_csv(f"{METRICS_DIR}pipe_mpi_metrics.csv", index=False)
    pipe_mpi_cluster_mean.to_csv(f"{METRICS_DIR}pipe_mpi_cluster_metrics.csv", index=False)
    
    # -------------- METRICS PLOTS ------------------------

    # plot_omp_metrics(seq_omp_mean, pipe_mean)
    # plot_mpi_metrics(pipe_mpi_mean, "1 node", f"{PLOTS_DIR}/mpi_local/")
    # plot_mpi_metrics(pipe_mpi_cluster_mean, "2 nodes", f"{PLOTS_DIR}/mpi_cluster/")

    # -------------- STAGE SERVICE TIMES PLOTS ------------------------

    plot_service_times(seq_omp_mean, "Sequential OMP Stage Service Times", f"{PLOTS_DIR}/seq_omp_service_time.png")
    plot_service_times(pipe_mean, "Pipeline OMP Stage Service Times", f"{PLOTS_DIR}/pipe_service_time.png")
    plot_service_times(pipe_mpi_mean, "MPI+OMP Total Stage Service Times (1 node)", f"{PLOTS_DIR}/mpi_local/mpi_service_time.png")
    plot_service_times(pipe_mpi_cluster_mean, "MPI+OMP Total Stage Service Times (2 nodes)", f"{PLOTS_DIR}/mpi_cluster/mpi_cluster_service_time.png")

