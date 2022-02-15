'''
benchmarking_algos.py
purpose: This file contains various TCR/CDR3 algorithms wrapped in a wrapper
         class which makes their APIs consistent. These wrapped versions of the
         algorithms will be used in the benchmarking.py script to compare the
         performances of various TCR/CDR3 algorithms in their ability to
         identify similarities between those that respond to the same epitope.
author: Yuta Nagano
version: 1.0.0
'''


# Imports
from abc import ABC, abstractmethod
import numpy as np
from numpy import dot
from numpy.linalg import norm
from polyleven import levenshtein
from source.atchley_encoder import atchley_encode


# Helper functions
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a,b) / (norm(a) * norm(b))


# Abstract base template class
class BenchmarkAlgo(ABC):
    '''
    All algorithms should be wrapped in such a way to always have two things: 1)
    a variable 'name' which should return a string that identifies what that
    algorithm is, and 2) a method 'similarity_func' which is a callable that
    takes two string arguments, cdr3_a and cdr3_b, and returns a numerical score
    that represents some similarity metric between the two.
    '''
    name = ''


    @staticmethod
    @abstractmethod
    def similarity_func(cdr3_a: str, cdr3_b: str):
        return 0
    

# Children classes
class NegativeLevenshtein(BenchmarkAlgo):
    # Negative levenshtein distance wrapped.
    name = 'Negative Levenshtein'


    @staticmethod
    def similarity_func(cdr3_a: str, cdr3_b: str) -> int:
        return -levenshtein(cdr3_a, cdr3_b)


class AtchleyCs(BenchmarkAlgo):
    '''
    This algorithm is one where for each cdr3, an 'average atchley factor' is
    calculated according to the constituent amino acid residues, and then a
    similarity score between two cdr3s are calculated as the cosine similarity
    between them. See source/atchley_encoder.py for more details.
    '''
    name = 'Averaged Atchley Factors + Cosine Distance'

    
    @staticmethod
    def similarity_func(cdr3_a: str, cdr3_b: str) -> float:
        return cosine_similarity(atchley_encode(cdr3_a), atchley_encode(cdr3_b))