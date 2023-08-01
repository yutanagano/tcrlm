from abc import ABC, abstractmethod
from enum import Enum
import re
from tidytcells import tcr


def get_v_gene_indices(gene_symbol):
    match = re.match(r"TR[AB]V([0-9]+)(-([0-9]+))?", gene_symbol)
    
    group_num = int(match.group(1))
    sub_num_if_any = 0 if match.group(3) is None else int(match.group(3))
    
    return (group_num, sub_num_if_any)


functional_travs = tcr.query(contains="TRAV", functionality="F", precision="gene")
functional_trbvs = tcr.query(contains="TRBV", functionality="F", precision="gene")


TravGene = Enum("Trav", sorted(functional_travs, key=get_v_gene_indices), module=__name__)
TrbvGene = Enum("Trbv", sorted(functional_trbvs, key=get_v_gene_indices), module=__name__)


class Tcrv(ABC):
    @property
    @abstractmethod
    def gene(self) -> str:
        pass

    @property
    @abstractmethod
    def allele_num(self) -> int:
        pass

    def get_cdr1_sequence(self) -> str:
        allele_symbol = self.__repr__()
        cdr1 = tcr.get_aa_sequence(allele_symbol)["CDR1-IMGT"]
        return cdr1

    def get_cdr2_sequence(self) -> str:
        allele_symbol = self.__repr__()
        cdr2 = tcr.get_aa_sequence(allele_symbol)["CDR2-IMGT"]
        return cdr2
    
    def __repr__(self) -> str:
        return f"{self.gene}*{self.allele_num:02d}"


class Trav(Tcrv):
    gene: str = None
    allele_num: int = None

    def __init__(self, gene: TravGene, allele_num: int) -> None:
        self.gene = gene
        self.allele_num = allele_num


class Trbv(Tcrv):
    gene: str = None
    allele_num: int = None

    def __init__(self, gene: TrbvGene, allele_num: int) -> None:
        self.gene = gene
        self.allele_num = allele_num


class Tcr:
    def __init__(self, trav: Trav, junction_a: str, trbv: Trbv, junction_b: str) -> None:
        self.trav = trav
        self.junction_a = junction_a
        self.trbv = trbv
        self.junction_b = junction_b