'''
This file contains various TCR/CDR3 algorithms wrapped in a wrapper class which
makes their APIs consistent. These wrapped versions of the algorithms will be
used in the benchmarking.py script to compare the performances of various
TCR/CDR3 algorithms in their ability to identify similarities between those
that respond to the same epitope.
'''


from abc import ABC, abstractmethod
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pathlib import Path
from polyleven import levenshtein
import torch
from typing import Union

from source.utils.atchleyencoder import atchley_encode
from source.utils.datahandling import tokenise
from source.utils.fileio import resolved_path_from_maybe_str


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a,b) / (norm(a) * norm(b))


class BenchmarkAlgo(ABC):
    '''
    All algorithms should be wrapped in such a way to always have two things: 1)
    a variable 'name' which should return a string that identifies what that
    algorithm is, and 2) a method 'similarity_func' which is a callable that
    takes two string arguments, cdr3_a and cdr3_b, and returns a numerical score
    that represents some similarity metric between the two.
    '''

    @property
    @abstractmethod
    def name(self) -> str: return ''


    @abstractmethod
    def similarity_func(self, cdr3_a: str, cdr3_b: str): return 0
    

# Children classes
class NegativeLevenshtein(BenchmarkAlgo):
    'Negative levenshtein distance wrapped.'

    @property
    def name(self) -> str:
        return 'Negative Levenshtein'


    def similarity_func(self, cdr3_a: str, cdr3_b: str) -> int:
        return -levenshtein(cdr3_a, cdr3_b)


class AtchleyCs(BenchmarkAlgo):
    '''
    This algorithm is one where for each cdr3, an 'average atchley factor' is
    calculated according to the constituent amino acid residues, and then a
    similarity score between two cdr3s are calculated as the cosine similarity
    between them. See source/atchley_encoder.py for more details.
    '''

    @property
    def name(self) -> str:
        return 'Averaged Atchley Factors + Cosine Distance'

    
    def similarity_func(self, cdr3_a: str, cdr3_b: str) -> float:
        return cosine_similarity(atchley_encode(cdr3_a), atchley_encode(cdr3_b))


class PretrainCdr3Bert(BenchmarkAlgo):
    'This is a wrapper to benchmark a pretrained instance of a Cdr3Bert model.'

    def __init__(
        self,
        path_to_model: Union[Path, str, None] = None,
        test_mode: bool = False
    ) -> None:
        # If in test mode, load demo model
        if test_mode:
            path_to_model = Path(
                'tests/resources/models/pretrained.ptnn'
            )
        
        # Otherwise load model specified by run id
        else:
            if path_to_model is None:
                raise RuntimeError(
                    'Please specify a path to a CDR3BERT model.')
            path_to_model = resolved_path_from_maybe_str(path=path_to_model)

        self.model = torch.load(path_to_model).bert.eval()

    
    @property
    def name(self) -> str:
        return 'CDR3 BERT (Pretrained)'


    @torch.no_grad()
    def similarity_func(self, cdr3_a: str, cdr3_b: str) -> float:
        cdr3_a_tokenised = tokenise(cdr3_a).unsqueeze(0)
        cdr3_b_tokenised = tokenise(cdr3_b).unsqueeze(0)
        return cosine_similarity(
            self.model.embed(cdr3_a_tokenised).squeeze().detach().numpy(),
            self.model.embed(cdr3_b_tokenised).squeeze().detach().numpy()
        )