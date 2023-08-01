from torch import Tensor


class VectorRepresentationDelegate:
    def get_vector_representations_of(self, tokenised_tcrs: Tensor, padding_mask: Tensor) -> Tensor:
        pass