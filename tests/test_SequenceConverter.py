import torch
import pytest
from data_handling import SequenceConverter


@pytest.fixture(scope='module')
def instantiate_converter_32_padding():
    converter = SequenceConverter(padding=32)
    yield converter


@pytest.fixture(scope='module')
def instantiate_converter_0_padding():
    converter = SequenceConverter(padding=0)
    yield converter


# Positive tests
@pytest.mark.parametrize(
    'aa,encoding',
    (
        ('A',torch.tensor([[-0.591,-1.302,-0.733,1.570,-0.146]], dtype=torch.float32)),
        ('H',torch.tensor([[0.336,-0.417,-1.673,-1.474,-0.078]], dtype=torch.float32)),
        ('M',torch.tensor([[-0.663,-1.524,2.219,-1.005,1.212]], dtype=torch.float32)),
        ('S',torch.tensor([[-0.228,1.399,-4.760,0.670,-2.647]], dtype=torch.float32)),
        ('W',torch.tensor([[-0.595,0.009,0.672,-2.128,-0.184]], dtype=torch.float32))
    )
)
def test_atchley_encodings(instantiate_converter_0_padding,aa,encoding):
    converter = instantiate_converter_0_padding
    assert(torch.equal(converter.to_atchley(aa), encoding))


@pytest.mark.parametrize(
    'aa,encoding',
    (
        ('A',torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],dtype=torch.float32)),
        ('H',torch.tensor([[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]],dtype=torch.float32)),
        ('M',torch.tensor([[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]],dtype=torch.float32)),
        ('S',torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]],dtype=torch.float32)),
        ('W',torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]],dtype=torch.float32))
    )
)
def test_one_hot_encodings(instantiate_converter_0_padding,aa,encoding):
    converter = instantiate_converter_0_padding
    assert(torch.equal(converter.to_one_hot(aa), encoding))


def test_atchley_sequence(instantiate_converter_0_padding):
    converter = instantiate_converter_0_padding

    sequence = 'AHMSW'
    expected = torch.tensor(
        [
            [-0.591,-1.302,-0.733,1.570,-0.146],
            [0.336,-0.417,-1.673,-1.474,-0.078],
            [-0.663,-1.524,2.219,-1.005,1.212],
            [-0.228,1.399,-4.760,0.670,-2.647],
            [-0.595,0.009,0.672,-2.128,-0.184]
        ],
        dtype=torch.float32
    )
    assert(torch.equal(converter.to_atchley(sequence),expected))


def test_onehot_sequence(instantiate_converter_0_padding):
    converter = instantiate_converter_0_padding

    sequence = 'AHMSW'
    expected = torch.tensor(
        [
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        ],
        dtype=torch.float32
    )
    assert(torch.equal(converter.to_one_hot(sequence),expected))


def test_atchley_padding(instantiate_converter_32_padding):
    converter = instantiate_converter_32_padding

    sequence = 'AHMSW'
    expected = torch.tensor(
        [
            [-0.591,-1.302,-0.733,1.570,-0.146],
            [0.336,-0.417,-1.673,-1.474,-0.078],
            [-0.663,-1.524,2.219,-1.005,1.212],
            [-0.228,1.399,-4.760,0.670,-2.647],
            [-0.595,0.009,0.672,-2.128,-0.184],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ],
        dtype=torch.float32
    )
    assert(torch.equal(converter.to_atchley(sequence),expected))


def test_onehot_padding(instantiate_converter_32_padding):
    converter = instantiate_converter_32_padding

    sequence = 'AHMSW'
    expected = torch.tensor(
        [
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ],
        dtype=torch.float32
    )
    assert(torch.equal(converter.to_one_hot(sequence),expected))


# Negative tests
def test_bad_sequence_conversion(instantiate_converter_0_padding):
    converter = instantiate_converter_0_padding

    with pytest.raises(RuntimeError):
        converter.to_atchley('This is not a valid amino acid sequence.')
    
    with pytest.raises(RuntimeError):
        converter.to_one_hot('This is not a valid amino acid sequence.')


def test_overly_long_sequence(instantiate_converter_32_padding):
    converter = instantiate_converter_32_padding

    with pytest.raises(RuntimeError):
        converter.to_atchley('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    with pytest.raises(RuntimeError):
        converter.to_one_hot('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')