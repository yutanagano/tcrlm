'''
training.py
purpose: Python module with helper classes for training CDR3Bert.
author: Yuta Nagano
ver: 1.1.0
'''


import torch


def create_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return x == 21


class ScheduledOptimiser:
    # Wrapper around optimiser to implement custom learning rate scheduling.
    def __init__(self,
                 optimiser: torch.optim.Optimizer,
                 lr_multiplier: float,
                 d_model: int,
                 n_warmup_steps: int):
        self.optimiser = optimiser
        self.lr_multiplier = lr_multiplier
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps

        self._step_num = 1


    @property
    def step_num(self) -> int:
        return self._step_num


    @property
    def lr(self) -> float:
        return self.calculate_lr(self._step_num)


    def step(self) -> None:
        # Update learning rate and step with the inner optimiser
        self._update_lr()
        self.optimiser.step()
        self._step_num += 1


    def zero_grad(self) -> None:
        # Zero out gradients with inner optimiser
        self.optimiser.zero_grad()
    

    def calculate_lr(self, step_num: int) -> float:
        # Learning rate scheduling
        return self.lr_multiplier * \
            self.d_model ** -0.5 * \
            min(step_num ** (-0.5), step_num * self.n_warmup_steps ** (-1.5))


    def _update_lr(self) -> None:
        # Update the learning rate of the inner optimiser
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = self.lr