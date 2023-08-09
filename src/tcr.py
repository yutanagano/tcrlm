from enum import Enum
import re
from tidytcells import tcr
from typing import Optional, Union


def get_v_gene_indices(gene_symbol):
    match = re.match(r"TR[AB]V([0-9]+)(-([0-9]+))?", gene_symbol)

    group_num = int(match.group(1))
    sub_num_if_any = 0 if match.group(3) is None else int(match.group(3))

    return (group_num, sub_num_if_any)


functional_travs = tcr.query(contains="TRAV", functionality="F", precision="gene")
functional_trbvs = tcr.query(contains="TRBV", functionality="F", precision="gene")


TravGene = Enum(
    "TravGene", sorted(functional_travs, key=get_v_gene_indices), module=__name__
)
TrbvGene = Enum(
    "TrbvGene", sorted(functional_trbvs, key=get_v_gene_indices), module=__name__
)


class Tcrv:
    def __init__(
        self, gene: Union[TravGene, TrbvGene, None], allele_num: Optional[int]
    ) -> None:
        self.gene = gene
        self.allele_num = allele_num

        if not self._gene_is_unknown() and self._allele_is_unknown():
            self._assume_first_allele()

    @property
    def cdr1_sequence(self) -> str:
        if self._gene_is_unknown():
            return None

        allele_symbol = self.__repr__()
        cdr1 = tcr.get_aa_sequence(allele_symbol)["CDR1-IMGT"]
        return cdr1

    @property
    def cdr2_sequence(self) -> str:
        if self._gene_is_unknown():
            return None

        allele_symbol = self.__repr__()
        cdr2 = tcr.get_aa_sequence(allele_symbol)["CDR2-IMGT"]
        return cdr2

    def __eq__(self, __value: object) -> bool:
        return self.gene == __value.gene and self.allele_num == __value.allele_num

    def __repr__(self) -> str:
        if self._gene_is_unknown():
            return "NA"

        return f"{self.gene.name}*{self.allele_num:02d}"

    def _gene_is_unknown(self):
        return self.gene is None
    
    def _allele_is_unknown(self):
        return self.allele_num is None
    
    def _assume_first_allele(self):
        self.allele_num = 1


class Tcr:
    def __init__(
        self, trav: Tcrv, junction_a_sequence: str, trbv: Tcrv, junction_b_sequence: str
    ) -> None:
        self._trav = trav
        self.junction_a_sequence = junction_a_sequence

        self._trbv = trbv
        self.junction_b_sequence = junction_b_sequence

    @property
    def cdr1a_sequence(self):
        return self._trav.cdr1_sequence

    @property
    def cdr2a_sequence(self):
        return self._trav.cdr2_sequence

    @property
    def cdr1b_sequence(self):
        return self._trbv.cdr1_sequence

    @property
    def cdr2b_sequence(self):
        return self._trbv.cdr2_sequence

    def __eq__(self, __value: object) -> bool:
        return (
            self._trav == __value._trav
            and self.junction_a_sequence == __value.junction_a_sequence
            and self._trbv == __value._trbv
            and self.junction_b_sequence == __value.junction_b_sequence
        )
