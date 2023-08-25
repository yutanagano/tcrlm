from typing import Optional


class MhcGene:
    def __init__(self, symbol: Optional[str]) -> None:
        self.symbol = symbol

    def __eq__(self, __value: object) -> bool:
        if self.symbol is None:
            return False
        
        self_is_subset_of_other = __value.symbol in self.symbol
        other_is_subset_of_self = self.symbol in __value.symbol

        return self_is_subset_of_other or other_is_subset_of_self
    
    def __repr__(self) -> str:
        if self.symbol is None:
            return "?"

        return self.symbol


class Pmhc:
    def __init__(self, epitope_sequence: Optional[str], mhc_a: MhcGene, mhc_b: MhcGene) -> None:
        self.epitope_sequence = epitope_sequence
        self.mhc_a = mhc_a
        self.mhc_b = mhc_b

    def __eq__(self, __value: object) -> bool:
        if self.epitope_sequence is None:
            return False
        
        same_epitope = (self.epitope_sequence == __value.epitope_sequence)
        same_mhc_a = (self.mhc_a == __value.mhc_a)
        same_mhc_b = (self.mhc_b == __value.mhc_b)

        return same_epitope and same_mhc_a and same_mhc_b
    
    def __repr__(self) -> str:
        epitope_representation = "?" if self.epitope_sequence is None else self.epitope_sequence
        return f"{self.mhc_a}, {self.mhc_b}, {epitope_representation}"