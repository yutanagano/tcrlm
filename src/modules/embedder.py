from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class Embedder(ABC, Module):
    '''
    Abstract base class for embedder modules.
    '''
    

    @property
    @abstractmethod
    def name(self) -> str:
        '''
        Return the name of the model as a string.
        '''


    @abstractmethod
    def embed(self, x: Tensor) -> Tensor:
        '''
        Given a tensor of tokenised TCRs, generate fixed-size vector embeddings
        for each of them.
        '''