import pytest
import torch
from pretrain import main as main_p
from finetune import main as main_f


# Positive tests
def test_pretrain():
    main_p('test', 'test', test_mode=True)
    # Test with gpu if available
    if torch.cuda.is_available():
        main_p('test', 'test', n_gpus=1, test_mode=True)
        # Test with multiple gpus if available
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            main_p('test', 'test', n_gpus=num_gpus, test_mode=True)


def test_finetune():
    main_f('test', 'test', test_mode=True)
    # Test with gpu if available
    if torch.cuda.is_available():
        main_f('test', 'test', n_gpus=1, test_mode=True)
        # Test with multiple gpus if available
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            main_f('test', 'test', n_gpus=num_gpus, test_mode=True)