import torch
from torch import Tensor
from typing import List, Tuple

from src.model.data.tokeniser.tokeniser import Tokeniser
from src.model.data.tokeniser.token_indices import AminoAcidTokenIndex, CdrCompartmentIndex
from src.tcr import Tcr


class BetaCdrTokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its beta chain CDRs 1 2 and 3.

    Dim 0: token index
    Dim 1: topen position
    Dim 2: CDR length
    Dim 3: CDR index
    """

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (AminoAcidTokenIndex.CLS, 0, 0, CdrCompartmentIndex.NULL)
        cdr1b = self._tokenise_cdr1b(tcr)
        cdr2b = self._tokenise_cdr2b(tcr)
        cdr3b = self._tokenise_cdr3b(tcr)

        all_cdrs_tokenised = [initial_cls_vector] + cdr1b + cdr2b + cdr3b

        return torch.tensor(all_cdrs_tokenised, dtype=torch.long)
    
    def tokenise_with_dropout(self, tcr: Tcr) -> Tensor:
        # TODO
        raise NotImplementedError
    
    def _tokenise_cdr1b(self, tcr: Tcr) -> List[Tuple[int]]:
        trbv = tcr.trbv
        cdr1_str = trbv.get_cdr1_sequence()
        return self._convert_to_numerical_form(cdr1_str, CdrCompartmentIndex.CDR1)
    
    def _tokenise_cdr2b(self, tcr: Tcr) -> List[Tuple[int]]:
        trbv = tcr.trbv
        cdr2_str = trbv.get_cdr2_sequence()
        return self._convert_to_numerical_form(cdr2_str, CdrCompartmentIndex.CDR2)
    
    def _tokenise_cdr3b(self, tcr: Tcr) -> List[Tuple[int]]:
        cdr3_str = tcr.junction_b
        return self._convert_to_numerical_form(cdr3_str, CdrCompartmentIndex.CDR3)
    
    def _convert_to_numerical_form(self, aa_sequence: str, cdr_index: CdrCompartmentIndex) -> List[Tuple[int]]:
        token_indices = [AminoAcidTokenIndex[aa] for aa in aa_sequence]
        token_positions = [idx for idx, _ in enumerate(aa_sequence, start=1)]
        cdr_length = [len(aa_sequence) for _ in aa_sequence]
        compartment_index = [cdr_index for _ in aa_sequence]
        
        iterator_over_token_vectors = zip(
            token_indices,
            token_positions,
            cdr_length,
            compartment_index
        )

        return list(iterator_over_token_vectors)