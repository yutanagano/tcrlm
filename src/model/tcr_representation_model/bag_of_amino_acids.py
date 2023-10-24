from abc import abstractmethod
from enum import Enum
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from src.model.tcr_representation_model import TcrRepresentationModel
from src import schema
from src.schema import AminoAcid, Tcr
from typing import List


class TcrChain(Enum):
    ALPHA = 0
    BETA = 1


class ModelScope(Enum):
    CDR3_ONLY = 0
    FULL = 1


class AbstractBagOfAminoAcids(TcrRepresentationModel):
    _one_hot_amino_acid_encodings = np.eye(len(AminoAcid))

    @abstractmethod
    def _chains_to_consider(self) -> List[TcrChain]:
        pass

    @abstractmethod
    def _scope(self) -> ModelScope:
        pass

    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        tcr_series = schema.generate_tcr_series(tcrs)
        tcr_vector_representations = tcr_series.map(self._represent_tcr_as_vector)
        return np.stack(tcr_vector_representations, axis=0)

    def _represent_tcr_as_vector(self, tcr: Tcr) -> ndarray:
        constituent_vectors = []

        if TcrChain.ALPHA in self._chains_to_consider:
            constituent_vectors.extend(self._represent_alpha_chain_as_vectors(tcr))
        if TcrChain.BETA in self._chains_to_consider:
            constituent_vectors.extend(self._represent_beta_chain_as_vectors(tcr))

        return np.sum(constituent_vectors, axis=0)

    def _represent_alpha_chain_as_vectors(self, tcr: Tcr) -> List[ndarray]:
        constituent_vectors = [
            self._represent_aa_sequence_as_vector(tcr.junction_a_sequence)
        ]

        if self._scope == ModelScope.FULL:
            constituent_vectors.extend(
                [
                    self._represent_aa_sequence_as_vector(tcr.cdr1a_sequence),
                    self._represent_aa_sequence_as_vector(tcr.cdr2a_sequence),
                ]
            )

        return constituent_vectors

    def _represent_beta_chain_as_vectors(self, tcr: Tcr) -> List[ndarray]:
        constituent_vectors = [
            self._represent_aa_sequence_as_vector(tcr.junction_b_sequence)
        ]

        if self._scope == ModelScope.FULL:
            constituent_vectors.extend(
                [
                    self._represent_aa_sequence_as_vector(tcr.cdr1b_sequence),
                    self._represent_aa_sequence_as_vector(tcr.cdr2b_sequence),
                ]
            )

        return constituent_vectors

    def _represent_aa_sequence_as_vector(self, aa_seq: str) -> ndarray:
        amino_acids_as_one_hot_encodings = list(
            map(self._amino_acid_as_one_hot_encoding, aa_seq)
        )
        return np.sum(amino_acids_as_one_hot_encodings, axis=0)

    def _amino_acid_as_one_hot_encoding(self, amino_acid_as_string: str) -> ndarray:
        amino_acid = AminoAcid[amino_acid_as_string]
        return self._one_hot_amino_acid_encodings[amino_acid.value]


class AlphaCdr3BagOfAminoAcids(AbstractBagOfAminoAcids):
    name = "Alpha CDR3 Bag of Amino Acids"
    distance_bins = range(25 + 1)
    _chains_to_consider = [TcrChain.ALPHA]
    _scope = ModelScope.CDR3_ONLY


class BetaCdr3BagOfAminoAcids(AbstractBagOfAminoAcids):
    name = "Beta CDR3 Bag of Amino Acids"
    distance_bins = range(25 + 1)
    _chains_to_consider = [TcrChain.BETA]
    _scope = ModelScope.CDR3_ONLY


class Cdr3BagOfAminoAcids(AbstractBagOfAminoAcids):
    name = "CDR3 Bag of Amino Acids"
    distance_bins = range(25 + 1)
    _chains_to_consider = [TcrChain.ALPHA, TcrChain.BETA]
    _scope = ModelScope.CDR3_ONLY


class AlphaBagOfAminoAcids(AbstractBagOfAminoAcids):
    name = "Alpha Bag of Amino Acids"
    distance_bins = range(25 + 1)
    _chains_to_consider = [TcrChain.ALPHA]
    _scope = ModelScope.FULL


class BetaBagOfAminoAcids(AbstractBagOfAminoAcids):
    name = "Beta Bag of Amino Acids"
    distance_bins = range(25 + 1)
    _chains_to_consider = [TcrChain.BETA]
    _scope = ModelScope.FULL


class BagOfAminoAcids(AbstractBagOfAminoAcids):
    name = "Bag of Amino Acids"
    distance_bins = range(25 + 1)
    _chains_to_consider = [TcrChain.ALPHA, TcrChain.BETA]
    _scope = ModelScope.FULL
