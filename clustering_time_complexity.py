import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from src.clustering import ClusteringMachine, BruteForceClusteringMachine, KdTreeClusteringMachine
from src.model import tcr_metric, tcr_representation_model
import statistics
import timeit
import torch


def main() -> None:
    time_complexity_stats = get_time_complexity_stats()
    save_results(time_complexity_stats)


def get_time_complexity_stats() -> dict:
    beta_cdr3_levenshtein = tcr_metric.BetaCdr3Levenshtein()
    beta_tcrdist = tcr_metric.BetaTcrdist()

    sceptr_cpu = tcr_representation_model.load_blastr_save(Path("model_saves/SCEPTR"), device="cpu")
    sceptr_cpu.name = "SCEPTR CPU"

    sceptr_gpu = tcr_representation_model.load_blastr_save(Path("model_saves/SCEPTR"), device=0)
    sceptr_gpu.name = "SCEPTR GPU"

    levenshtein_bf = BruteForceClusteringMachine(beta_cdr3_levenshtein)
    tcrdist_bf = BruteForceClusteringMachine(beta_tcrdist)
    sceptr_cpu_bf = BruteForceClusteringMachine(sceptr_cpu)
    sceptr_cpu_kdt = KdTreeClusteringMachine(sceptr_cpu)
    sceptr_gpu_bf = BruteForceClusteringMachine(sceptr_gpu)
    sceptr_gpu_kdt = KdTreeClusteringMachine(sceptr_gpu)

    stats = dict()

    for model in (
        levenshtein_bf,
        tcrdist_bf,
        sceptr_cpu_bf,
        sceptr_gpu_bf,
        sceptr_cpu_kdt,
        sceptr_gpu_kdt
    ):
        model_stats = get_time_complexity_stats_for_model(model)
        stats[model.name] = model_stats

    return stats


def get_time_complexity_stats_for_model(model: ClusteringMachine) -> dict:
    print(f"Benchmarking {model.name}...")

    data = load_tcr_data()
    model_stats = dict()

    for num_tcrs in np.logspace(1, 5, num=10, dtype=int):
        num_repeats = math.ceil(10_000 / num_tcrs)

        print(f"Benchmarking {num_tcrs} TCRs...")
        repeated_times = timeit.repeat(lambda: model.cluster(data[:num_tcrs], 1), number=1, repeat=num_repeats)
        time_in_seconds = statistics.mean(repeated_times)
        print(f"{time_in_seconds} seconds.")
        model_stats[int(num_tcrs)] = time_in_seconds

    return model_stats


def load_tcr_data() -> DataFrame:
    data = pd.read_csv("tcr_data/preprocessed/olga/olga.csv")
    data.TRAV = data.TRAV.map(lambda x: x+"*01")
    data.TRBV = data.TRBV.map(lambda x: x+"*01")
    return data


def save_results(stats: dict) -> None:
    with open("time_complexity_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
