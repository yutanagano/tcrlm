import pandas as pd
from pandas import DataFrame, Series
from torch import Tensor
from torch.utils.data import Dataset

from src.data.tcr import Tcr, Tcrv, TravGene, TrbvGene
from src.data.tokeniser.tokeniser import Tokeniser


class TcrDataset(Dataset):
    def __init__(self, data: DataFrame, tokeniser: Tokeniser):
        super().__init__()
        self._tcr_series = self._generate_tcr_series_from(data)
        self._tokeniser = tokeniser

    def __len__(self) -> int:
        return len(self._tcr_series)

    def __getitem__(self, index: int) -> Tensor:
        tcr_at_index = self._tcr_series.iloc[index]
        return self._tokenise(tcr_at_index)

    def _generate_tcr_series_from(self, data: DataFrame) -> Series:
        tcr_series = data.apply(self._generate_tcr_from_row, axis="columns")
        return tcr_series

    def _generate_tcr_from_row(self, row: Series) -> Tcr:
        trav = self._get_trav_object_from_symbol(row.TRAV)
        trbv = self._get_trbv_object_from_symbol(row.TRBV)

        junction_a = self._get_value_if_not_na_else_none(row.CDR3A)
        junction_b = self._get_value_if_not_na_else_none(row.CDR3B)

        return Tcr(trav, junction_a, trbv, junction_b)

    def _get_trav_object_from_symbol(self, trav_symbol: str) -> Tcrv:
        if pd.isna(trav_symbol):
            return Tcrv(None, None)

        gene = self._get_trav_gene_object_from(trav_symbol)
        allele_number = self._get_allele_number_from(trav_symbol)

        return Tcrv(gene, allele_number)

    def _get_trbv_object_from_symbol(self, trbv_symbol: str) -> Tcrv:
        if pd.isna(trbv_symbol):
            return Tcrv(None, None)

        gene = self._get_trbv_gene_object_from(trbv_symbol)
        allele_number = self._get_allele_number_from(trbv_symbol)

        return Tcrv(gene, allele_number)

    def _get_trav_gene_object_from(self, symbol: str) -> TravGene:
        str_representing_gene = symbol.split("*")[0]
        return TravGene[str_representing_gene]

    def _get_trbv_gene_object_from(self, symbol: str) -> TrbvGene:
        str_representing_gene = symbol.split("*")[0]
        return TrbvGene[str_representing_gene]

    def _get_allele_number_from(self, symbol: str) -> int:
        split_at_asterisk = symbol.split("*")
        has_allele_number = len(split_at_asterisk) == 2

        if not has_allele_number:
            return None

        str_representing_allele_number = split_at_asterisk[1]

        return int(str_representing_allele_number)

    def _get_value_if_not_na_else_none(self, value) -> any:
        if pd.isna(value):
            return None

        return value

    def _tokenise(self, tcr: Tcr) -> Tensor:
        return self._tokeniser.tokenise(tcr)
