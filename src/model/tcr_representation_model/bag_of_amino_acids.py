from numpy import ndarray
from pandas import DataFrame
from src.model.tcr_representation_model import TcrRepresentationModel


class BetaCdr3BagOfAminoAcids(TcrRepresentationModel):
    def calc_vector_representations(self, tcrs: DataFrame) -> ndarray:
        beta_cdr3s = tcrs.CDR3B
