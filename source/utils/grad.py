'Custom utilities for gradient descent.'


import torch
from typing import Tuple


class AdamWithScheduling:
    'Wrapper around optimiser to implement custom learning rate scheduling.'

    def __init__(
        self,
        params,
        d_model: int,
        n_warmup_steps: int,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        lr_multiplier: float = 1,
        scheduling: bool = True,
        decay: bool = True
    ):
        self.optimiser = torch.optim.Adam(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps
        )
        self._lr_multiplier = lr_multiplier
        self._d_model = d_model
        self._n_warmup_steps = n_warmup_steps
        self._scheduling = scheduling
        self._decay = decay

        self._step_num = 1
        self._lr_explicit = lr


    @property
    def step_num(self) -> int:
        return self._step_num


    @property
    def lr(self) -> float:
        if self._scheduling:
            return self.calculate_lr(self._step_num)
        
        return self._lr_explicit


    def step(self) -> None:
        # Update learning rate and step with the inner optimiser
        if self._scheduling:
            self._update_lr()
        
        self.optimiser.step()
        self._step_num += 1


    def zero_grad(self) -> None:
        self.optimiser.zero_grad()
    

    def calculate_lr(self, step_num: int) -> float:
        # Learning rate decays inversely with the square root of step number
        if self._decay:
            return self._lr_multiplier * self._d_model ** -0.5 * \
                min(
                    step_num ** (-0.5),
                    step_num * self._n_warmup_steps ** (-1.5)
                )
        # Learning rate reaches target and stays there
        return min(
            self._lr_explicit,
            step_num / self._n_warmup_steps * self._lr_explicit
        )


    def _update_lr(self) -> None:
        # Update the learning rate of the inner optimiser
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = self.lr