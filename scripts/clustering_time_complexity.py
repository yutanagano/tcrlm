from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import pandas as pd
from src.clustering import BruteForceClusteringMachine, KdTreeClusteringMachine
from src.model import tcr_representation_model


def main() -> None:
    time_complexity_stats = get_time_complexity_stats()
    fig = plot(time_complexity_stats)
    fig.savefig("time_complexity_stats.png")


def get_time_complexity_stats() -> dict:
    blastr = tcr_representation_model.load_blastr_save("some/path")
    blastr_brute_force = BruteForceClusteringMachine(blastr)
    blastr_kd_tree = KdTreeClusteringMachine(blastr)

    data = pd.read_csv("some/path")

    for subsample_size in (100, 1000, 10_000, 100_000, )


def plot(stats: dict) -> Figure:
    pass


if __name__ == "__main__":
    main()
