import torch


class MockDistributedDataParallel:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    
    def parameters(self):
        return self.module.parameters()