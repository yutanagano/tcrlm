import itertools
import Levenshtein
import math
from numpy import ndarray
import numpy as np
from pandas import DataFrame, Series
from pathlib import Path
import pickle
from scipy.spatial import distance
from src.model.tcr_metric import TcrMetric
from typing import Dict


def _load_pmats() -> Dict[int, ndarray]:
    pmats = dict()

    for i in range(1, 9):
        pmats[i] = _loadpic(f"PMATs_dL{i}")
    
    return pmats


def _load_vlograts() -> Dict[int, ndarray]:
    vlograts = dict()

    for i in range(1, 9):
        vlograts[i] = _loadpic(f"VlogRat_L{i}")

    return vlograts


def _loadpic(filename: str) -> ndarray:
    path_to_file = Path(__file__).parent/"Params"/filename

    with open(path_to_file, 'rb') as f:
        a = pickle.load(f)
    return a


class BetaMlTcrDist(TcrMetric):
    name = "Beta ML tcrdist"
    distance_bins = range(100)

    _PMATS = _load_pmats()
    _VLOGRATS = _load_vlograts()
    _AA_TO_NUMBER = {c: i for i, c in enumerate('CMFILVWYAGTSNQDEHRKP-')}
    _V_TO_NUMBER = {c: i for i, c in enumerate(_loadpic("Vref"))}

    def calc_cdist_matrix(self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame) -> ndarray:
        packaged_anchor_tcrs = anchor_tcrs.apply(
            self._package_sequence, axis=1
        )
        packaged_comparison_tcrs = comparison_tcrs.apply(
            self._package_sequence, axis=1
        )

        num_anchors = len(anchor_tcrs)
        num_comparisons = len(comparison_tcrs)

        cdist_matrix = np.empty((num_anchors, num_comparisons), dtype=float)

        for anchor_index, comparison_index in itertools.product(range(num_anchors), range(num_comparisons)):
            cdist_matrix[anchor_index, comparison_index] = self._compute_distance(
                packaged_anchor_tcrs.iloc[anchor_index],
                packaged_comparison_tcrs.iloc[comparison_index]
            )

        return cdist_matrix
    
    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(tcrs, tcrs)
        return distance.squareform(pdist_matrix, checks=False)
    
    @staticmethod
    def _package_sequence(row: Series) -> tuple:
        trbv: str = row.TRBV

        if "*" in trbv:
            trbv = trbv.split("*")[0]

        return (row.CDR3B, trbv)
    
    def _compute_distance(self, anchor_tcr: tuple, comparison_tcr: tuple) -> float:
        dr = Levenshtein.distance(anchor_tcr[0],comparison_tcr[0])
        JJJ = [np.log(8e-2), np.log(1e-2), np.log(3e-3),np.log(1e-3)]
        d = 0
        if dr > 0.5 and dr < 8.5:
            mat = self._nmat_gen([[anchor_tcr[0],comparison_tcr[0],(self._V_TO_NUMBER[anchor_tcr[1]],self._V_TO_NUMBER[comparison_tcr[1]])]],dr)
            d = self._distance_v(mat,dr)[0]
        if math.isnan(d):
            d = 0
        if dr < 3.5:
            d = d - JJJ[dr-1]
        if dr > 3.5:
            d = dr - JJJ[3] 
        return d
    
    def _nmat_gen(self, pairs2, dmax) -> ndarray:
        pairs = pairs2
        matrix1 = -1*np.ones((len(pairs),4*dmax + 2),dtype = int)
        for ind1, (s1, s2, Vg) in enumerate(pairs):
            steps = Levenshtein.editops(s1,s2)
            LOP = len(s1)
            Vg1, Vg2 = sorted(Vg)
            matrix1[ind1, 0] = int(Vg1)
            matrix1[ind1, 1] = int(Vg2)
            for ind, (op, i1, i2) in enumerate(steps):
                if op == 'replace': # 1 is replace
                    ii1, ii2 = sorted([self._AA_TO_NUMBER[s1[i1]],self._AA_TO_NUMBER[s2[i2]]])
                    matrix1[ind1,(2+4*ind):(2+4*(ind+1))] = [LOP, i1, ii1, ii2]
                if op == 'delete': # 0 is delete or insert
                    ii1, ii2 = sorted([self._AA_TO_NUMBER[s1[i1]],20]) # 20 is blank
                    matrix1[ind1,(2+4*ind):(2+4*(ind+1))] = [LOP, i1, ii1, ii2]
                    LOP += -1
                if op == 'insert': # 0 is delete or insert
                    ii1, ii2 = sorted([self._AA_TO_NUMBER[s2[i2]],20]) # 20 is blank
                    matrix1[ind1,(2+4*ind):(2+4*(ind+1))] = [LOP, i1, ii1, ii2]
                    LOP += 1
        matrix = []
        matrix.extend(matrix1)
        return np.array(matrix)
    
    def _distance_v(self, matrix, dL) -> ndarray:
        Vg1 = matrix[:,0]
        Vg2 = matrix[:,1]
        L = matrix[:,2::4]
        pos = matrix[:,3::4]
        i1 = matrix[:,4::4]
        i2 = matrix[:,5::4]
        a = self._PMATS[dL]
        AW = a[0]
        PW = a[1]
        bard = a[2]
        VR = self._VLOGRATS[dL]
        c5 = VR[Vg1,Vg2]
        for ii in range(dL):
            if ii == 0:
                a = np.exp(PW[pos[:,ii], L[:,ii]])
                b = np.exp(AW[i1[:,ii], i2[:,ii]])
            else:
                a = a + np.exp(PW[pos[:,ii], L[:,ii]])
                b = b + np.exp(AW[i1[:,ii], i2[:,ii]])
        return np.log(a/dL) + np.log(b/dL) + c5 + bard