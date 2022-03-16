import pytest
from torch.nn import Linear
from source.training import AdamWithScheduling


@pytest.fixture(scope='function')
def instantiate_scheduled_optimiser():
    model = Linear(4, 4)
    scheduled_optim = AdamWithScheduling(
        params=model.parameters(),
        d_model=4,
        n_warmup_steps=5,
        lr_multiplier=2
    )
    yield scheduled_optim


# Positive tests
def test_step_num(instantiate_scheduled_optimiser):
    optim = instantiate_scheduled_optimiser

    for i in range(1,1+5):
        assert(optim.step_num == i)
        optim.zero_grad()
        optim.step()


def test_scheduled_lr(instantiate_scheduled_optimiser):
    optim = instantiate_scheduled_optimiser

    def calculate_lr(step_num):
        return 2 * (4 ** -0.5) * \
            min((step_num ** -0.5), (step_num * (5 ** -1.5)))
    
    for i in range(1,1+10):
        assert(optim.lr == calculate_lr(i))
        optim.zero_grad()
        optim.step()
        
        for param_group in optim.optimiser.param_groups:
            assert(param_group['lr'] == calculate_lr(i))


def test_scheduled_lr_no_decay():
    model = Linear(4, 4)
    optim = AdamWithScheduling(
        params=model.parameters(),
        d_model=4,
        n_warmup_steps=5,
        decay=False
    )

    def calculate_lr(step_num):
        return min(
            0.001,
            step_num / 5 * 0.001
        )
    
    for i in range(1,1+10):
        assert(optim.lr == calculate_lr(i))
        optim.zero_grad()
        optim.step()

        for param_group in optim.optimiser.param_groups:
            assert(param_group['lr'] == calculate_lr(i))


def test_unscheduled_lr():
    model = Linear(4, 4)
    optim = AdamWithScheduling(
        params=model.parameters(),
        d_model=4,
        n_warmup_steps=5,
        scheduling=False
    )

    for i in range(10):
        assert(optim.lr == 0.001)
        optim.zero_grad()
        optim.step()
        
        for param_group in optim.optimiser.param_groups:
            assert(param_group['lr'] == 0.001)


# Negative tests
def test_set_step_num(instantiate_scheduled_optimiser):
    optim = instantiate_scheduled_optimiser
    
    with pytest.raises(AttributeError):
        optim.step_num = 10


def test_set_lr(instantiate_scheduled_optimiser):
    optim = instantiate_scheduled_optimiser

    with pytest.raises(AttributeError):
        optim.lr = 0.1