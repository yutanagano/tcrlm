"""
tcrdist edition of benchmarking pipelines.
"""


from numpy import ndarray
from .pure_metric_benchmarking_pipeline import PureMetricBenchmarkingPipeline
import numpy as np
from pathlib import Path
from pandas import DataFrame
import pandas as pd
from tcrdist.repertoire import TCRrep
from tcrdist.rep_funcs import _pws
from tcrdist import repertoire_db
import pwseqdist as pw
from tqdm import tqdm
import warnings


all_genes = repertoire_db.RefGeneSet('alphabeta_gammadelta_db.tsv').all_genes


def _map_gene_to_reference_seq2(
                                organism,
                                gene,
                                cdr,
                                attr = 'cdrs_no_gaps'):
    """
    Taken and modified from tcrdist.repertoire.TCRrep
    """
    try:
        aa_string = all_genes[organism][gene].__dict__[attr][cdr]
    except KeyError:
        aa_string = None
        warnings.warn("{} gene was not recognized in reference db no cdr seq could be inferred".format(gene), stacklevel=2)
    return(aa_string)


def infer_cdrs_from_v_gene(cell_df, chain, organism = "human", imgt_aligned = True):
    """
    Taken and modified from tcrdist.repertoire.TCRrep
    """

    if not imgt_aligned:
        f0 = lambda v : _map_gene_to_reference_seq2(gene = v,
                                                            cdr = 0,
                                                            organism = organism,
                                                            attr ='cdrs_no_gaps')
        f1 = lambda v : _map_gene_to_reference_seq2(gene = v,
                                                            cdr = 1,
                                                            organism = organism,
                                                            attr ='cdrs_no_gaps')
        f2 = lambda v : _map_gene_to_reference_seq2(gene = v,
                                                            cdr = 2,
                                                            organism = organism,
                                                            attr ='cdrs_no_gaps')
    else:
        imgt_aligned_status = True
        f0 = lambda v : _map_gene_to_reference_seq2(gene = v,
                                                            cdr = 0,
                                                            organism = organism,
                                                            attr ='cdrs')
        f1 = lambda v : _map_gene_to_reference_seq2(gene = v,
                                                            cdr = 1,
                                                            organism = organism,
                                                            attr ='cdrs')
        f2 = lambda v : _map_gene_to_reference_seq2(gene = v,
                                                            cdr = 2,
                                                            organism = organism,
                                                            attr ='cdrs')
    if chain == "alpha":
        cell_df = cell_df.assign(cdr1_a_aa=list(map(f0, cell_df.v_a_gene)),
                                            cdr2_a_aa=list(map(f1, cell_df.v_a_gene)),
                                            pmhc_a_aa=list(map(f2, cell_df.v_a_gene)))
    if chain == "beta":
        cell_df = cell_df.assign(cdr1_b_aa=list(map(f0, cell_df.v_b_gene)),
                                            cdr2_b_aa=list(map(f1, cell_df.v_b_gene)),
                                            pmhc_b_aa=list(map(f2, cell_df.v_b_gene)))
    if chain == "gamma":
        cell_df = cell_df.assign(cdr1_g_aa=list(map(f0, cell_df.v_g_gene)),
                                            cdr2_g_aa=list(map(f1, cell_df.v_g_gene)),
                                            pmhc_g_aa=list(map(f2, cell_df.v_g_gene)))
    if chain == "delta":
        cell_df = cell_df.assign(cdr1_d_aa=list(map(f0, cell_df.v_d_gene)),
                                            cdr2_d_aa=list(map(f1, cell_df.v_d_gene)),
                                            pmhc_d_aa=list(map(f2, cell_df.v_d_gene)))

    return cell_df

def get_pws_kwargs(chain: str) -> dict:
    metrics = { 
                f"cdr3_{chain}_aa" : pw.metrics.nb_vector_tcrdist,
                f"pmhc_{chain}_aa" : pw.metrics.nb_vector_tcrdist,
                f"cdr2_{chain}_aa" : pw.metrics.nb_vector_tcrdist,
                f"cdr1_{chain}_aa" : pw.metrics.nb_vector_tcrdist }
    weights = { 
                f"cdr3_{chain}_aa" : 3,
                f"pmhc_{chain}_aa" : 1,
                f"cdr2_{chain}_aa" : 1,
                f"cdr1_{chain}_aa" : 1}
    kargs = {
                f"cdr3_{chain}_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':False},
                f"pmhc_{chain}_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
                f"cdr2_{chain}_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
                f"cdr1_{chain}_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True}}
    return {
        "metrics": metrics, "weights": weights, "kargs": kargs}


class TcrdistBenchmarkingPipeline(PureMetricBenchmarkingPipeline):
    MODEL_NAME = "tcrdist"
    CHAINS = ["alpha", "beta"]

    @staticmethod
    def load_csv(path: Path) -> DataFrame:
        df = pd.read_csv(path)
        df = df.rename(
            columns={
                "TRAV": "v_a_gene",
                "CDR3A": "cdr3_a_aa",
                "TRBV": "v_b_gene",
                "CDR3B": "cdr3_b_aa",
                "duplicate_count": "count",
            }
        )
        for column in ["v_a_gene", "v_b_gene"]:
            if column in df:
                df[column] = df[column].map(
                    lambda x: x if not type(x) == str or "*" in x else x + "*01"
                )
        if not "count" in df:
            df["count"] = 1
        return df

    def get_pdist_matrix(cls, ds_name: str, ds_df: DataFrame) -> ndarray:
        tr = TCRrep(cell_df=ds_df, organism="human", chains=cls.CHAINS, deduplicate=False)

        pdist_matrix = np.zeros((len(ds_df), len(ds_df)), dtype=np.float32)

        if "alpha" in cls.CHAINS:
            pdist_matrix += tr.pw_alpha.astype(np.float32)
        if "beta" in cls.CHAINS:
            pdist_matrix += tr.pw_beta.astype(np.float32)

        return pdist_matrix

    def get_avg_dist_to_100nn_over_background(cls) -> ndarray:
        background_nona = cls.background_data.dropna(subset=["v_a_gene", "cdr3_a_aa", "v_b_gene", "cdr3_b_aa"])

        avg_dists = []

        for i in tqdm(range(0, len(background_nona), 100)):
            dists_batch = cls.get_cdist_matrix(
                background_nona.iloc[i:i+100],
                background_nona
            ).squeeze()

            for dists in dists_batch:
                dists_to_closest_100 = np.partition(dists, kth=100)[:100]
                avg_dist = dists_to_closest_100.mean()
                avg_dists.append(avg_dist)

        return np.array(avg_dists, dtype=np.float32)
    
    def get_cdist_matrix(cls, ds_a_df: DataFrame, ds_b_df: DataFrame) -> ndarray:
        for chain in cls.CHAINS:
            ds_a_df = infer_cdrs_from_v_gene(ds_a_df, chain)
            ds_b_df = infer_cdrs_from_v_gene(ds_b_df, chain)

        cdist_matrix = np.zeros((len(ds_a_df), len(ds_b_df)), dtype=np.float32)

        if "alpha" in cls.CHAINS:
            cdist_matrix += _pws(df=ds_a_df, df2=ds_b_df, **get_pws_kwargs("a"))["tcrdist"].astype(np.float32)
        if "beta" in cls.CHAINS:
            cdist_matrix += _pws(df=ds_a_df, df2=ds_b_df, **get_pws_kwargs("b"))["tcrdist"].astype(np.float32)
        
        return cdist_matrix

class BTcrdistBenchmarkingPipeline(TcrdistBenchmarkingPipeline):
    CHAINS = ["beta"]
