import re
import torch
from typing import Union


class MockDevice:
    def __init__(self, device_id: Union[str, int]) -> None:
        if type(device_id) == int:
            self.type = 'cuda'
            self.index = device_id
            return
        
        if not type(device_id) == str:
            raise RuntimeError(
                'device_id must be of type str or int. '
                f'Got {type(device_id)}.'
            )
        
        if re.match('cuda:[0-9]+', device_id):
            self.type = 'cuda'
            self.index = int(re.search('[0-9]+', device_id)[0])
            return
        
        if device_id == 'cpu':
            self.type = 'cpu'
            self.index = None
            return

        raise RuntimeError(f'device_id not recognised: {device_id}')


    def __repr__(self) -> str:
        if self.type == 'cuda':
            return f"device(type='cuda', index={self.index})"
        
        return "device(type='cpu')"


    def __str__(self) -> str:
        if self.type == 'cuda':
            return f'cuda:{self.index}'

        return 'cpu'


class MockDistributedDataParallel:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    
    def parameters(self):
        return self.module.parameters()