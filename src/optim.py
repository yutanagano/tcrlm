"""
Custom optimisers
"""


import torch


class AdamWithScheduling:
    "Wrapper around optimiser to implement custom learning rate scheduling."

    def __init__(
        self,
        params,
        d_model: int,
        n_warmup_steps: int,
        lr_multiplier: float = 1,
        **kwargs
    ):
        self.optimiser = torch.optim.Adam(params=params, **kwargs)
        self._d_model = d_model
        self._n_warmup_steps = n_warmup_steps
        self._lr_multiplier = lr_multiplier

        self._step_num = 1

    @property
    def lr(self) -> float:
        return self.calculate_lr(self._step_num)

    def step(self) -> None:
        """
        Update learning rate and step with the inner optimiser
        """

        self._update_lr()
        self.optimiser.step()
        self._step_num += 1

    def zero_grad(self) -> None:
        self.optimiser.zero_grad()

    def calculate_lr(self, step_num: int) -> float:
        # Learning rate decays inversely with the square root of step number
        return (
            self._lr_multiplier
            * self._d_model**-0.5
            * min(step_num ** (-0.5), step_num * self._n_warmup_steps ** (-1.5))
        )

    def _update_lr(self) -> None:
        # Update the learning rate of the inner optimiser
        for param_group in self.optimiser.param_groups:
            param_group["lr"] = self.lr
