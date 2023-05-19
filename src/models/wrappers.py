"""
Model wrappers to export default forward procedure for a particular training.
"""


from .embedder import _MLMEmbedder
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class ModelWrapper(Module, ABC):
    def __init__(self, embedder: _MLMEmbedder) -> None:
        super().__init__()
        self.embedder = embedder

    @abstractmethod
    def forward(self) -> Tensor:
        pass


class MLMModelWrapper(ModelWrapper):
    def forward(self, masked: Tensor) -> Tensor:
        return self.embedder.mlm(masked)


class CLModelWrapper(ModelWrapper):
    def forward(self, x: Tensor, x_prime: Tensor, masked: Tensor) -> tuple:
        z = self.embedder.embed(x)
        z_prime = self.embedder.embed(x_prime)
        mlm_logits = self.embedder.mlm(masked)

        return z, z_prime, mlm_logits


class CombinedCLModelWrapper(ModelWrapper):
    def forward(
        self, bg: Tensor, bg_prime: Tensor, ep: Tensor, ep_prime: Tensor, masked: Tensor
    ) -> tuple:
        bg_z = self.embedder.embed(bg)
        bg_prime_z = self.embedder.embed(bg_prime)

        ep_z = self.embedder.embed(ep)
        ep_prime_z = self.embedder.embed(ep_prime)

        mlm_logits = self.embedder.mlm(masked)

        return bg_z, bg_prime_z, ep_z, ep_prime_z, mlm_logits
