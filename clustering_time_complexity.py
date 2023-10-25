import json
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from src.clustering import ClusteringMachine, BruteForceClusteringMachine, KdTreeClusteringMachine
from src.model import tcr_metric, tcr_representation_model
import timeit


def main() -> None:
    time_complexity_stats = get_time_complexity_stats()
    save_results(time_complexity_stats)


def get_time_complexity_stats() -> dict:
    beta_cdr_levenshtein = tcr_metric.BetaCdrLevenshtein()
    beta_tcrdist = tcr_metric.BetaTcrdist()
    blastr = tcr_representation_model.load_blastr_save(Path("/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/model_saves/Beta_CDR_BERT_Unsupervised_Large_2"))

    levenshtein_bf = BruteForceClusteringMachine(beta_cdr_levenshtein)
    tcrdist_bf = BruteForceClusteringMachine(beta_tcrdist)
    blastr_bf = BruteForceClusteringMachine(blastr)
    blastr_kdt = KdTreeClusteringMachine(blastr)

    stats = dict()

    for model in (
        levenshtein_bf,
        tcrdist_bf,
        blastr_bf,
        blastr_kdt
    ):
        model_stats = get_time_complexity_stats_for_model(model)
        stats[model.name] = model_stats

    return stats


def get_time_complexity_stats_for_model(model: ClusteringMachine) -> dict:
    data = load_tcr_data()
    model_stats = dict()

    for num_tcrs in (100, 500, 1_000, 5_000, 10_000, 50_000, 100_000):
        time_in_seconds = timeit.timeit(lambda: model.cluster(data[:num_tcrs], 1), number=1)
        model_stats[num_tcrs] = time_in_seconds

    return model_stats


def load_tcr_data() -> DataFrame:
    data = pd.read_csv("/home/yutanagano/UCLOneDrive/MBPhD/projects/tcr_embedder/data/preprocessed/tanno/test.csv")
    data.TRAV = data.TRAV.map(lambda x: x+"*01")
    data.TRBV = data.TRBV.map(lambda x: x+"*01")
    return data


def save_results(stats: dict) -> None:
    with open("time_complexity_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
