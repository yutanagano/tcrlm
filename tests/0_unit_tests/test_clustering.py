from src import clustering
from src.model.tcr_representation_model import BagOfAminoAcids


def test_brute_force_vs_kd_tree(mock_data_df):
    THRESHOLD = 7

    tcr_model = BagOfAminoAcids()
    brute_force = clustering.BruteForceClusteringMachine(tcr_model)
    kd_tree = clustering.KdTreeClusteringMachine(tcr_model)

    brute_force_results = brute_force.cluster(mock_data_df, THRESHOLD)
    kd_tree_results = kd_tree.cluster(mock_data_df, THRESHOLD)

    assert brute_force_results == kd_tree_results
