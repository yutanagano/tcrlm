import pytest
from src.testing import mockups
import torch


class TestMockDevice:
    def test_cpu(self):
        device = mockups.MockDevice('cpu')

        assert device.type == 'cpu'
        assert device.index == None
        assert repr(device) == "device(type='cpu')"
        assert str(device) == 'cpu'


    @pytest.mark.parametrize(
        'index', (0, 1)
    )
    def test_cuda_string(self, index):
        device = mockups.MockDevice(f'cuda:{index}')

        assert device.type == 'cuda'
        assert device.index == index
        assert repr(device) == f"device(type='cuda', index={index})"
        assert str(device) == f'cuda:{index}'


    @pytest.mark.parametrize(
        'index', (0, 1)
    )
    def test_cuda_int(self, index):
        device = mockups.MockDevice(index)

        assert device.type == 'cuda'
        assert device.index == index
        assert repr(device) == f"device(type='cuda', index={index})"
        assert str(device) == f'cuda:{index}'


    def test_error_bad_type(self):
        with pytest.raises(RuntimeError):
            mockups.MockDevice(['foo', 'bar'])


    def test_error_bad_string(self):
        with pytest.raises(RuntimeError):
            mockups.MockDevice('foobar')


class TestMockDistributedDataParallel:
    def test_mock_distributed_data_parallel(self):
        model = torch.nn.Linear(3,3)
        ddp = mockups.MockDistributedDataParallel(module=model)
        parameters_zip = zip(ddp.parameters(), model.parameters())

        assert ddp.module == model
        for p1, p2 in parameters_zip:
            assert torch.equal(p1, p2)