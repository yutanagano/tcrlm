import pytest
from torch.nn import Linear
from torch.optim import Adam
from source.training import ScheduledOptimiser


@pytest.fixture(scope='function')
def instantiate_optimiser():
    model = Linear(4, 4)
    optim = Adam(model.parameters())
    scheduled_optim = ScheduledOptimiser(optimiser=optim,
                                         lr_multiplier=2,
                                         d_model=4,
                                         n_warmup_steps=5)
    yield scheduled_optim


# Positive tests
def test_step_num(instantiate_optimiser):
    optim = instantiate_optimiser

    for i in range(1,1+5):
        assert(optim.step_num == i)
        optim.zero_grad()
        optim.step()


def test_lr(instantiate_optimiser):
    optim = instantiate_optimiser

    def calculate_lr(step_num):
        return 2 * (4 ** -0.5) * \
            min((step_num ** -0.5), (step_num * (5 ** -1.5)))
    
    for i in range(1,1+10):
        assert(optim.lr == calculate_lr(i))
        optim.zero_grad()
        optim.step()
        
        for param_group in optim.optimiser.param_groups:
            assert(param_group['lr'] == calculate_lr(i))


# Negative tests
def test_set_step_num(instantiate_optimiser):
    optim = instantiate_optimiser
    
    with pytest.raises(AttributeError):
        optim.step_num = 10


def test_set_lr(instantiate_optimiser):
    optim = instantiate_optimiser

    with pytest.raises(AttributeError):
        optim.lr = 0.1