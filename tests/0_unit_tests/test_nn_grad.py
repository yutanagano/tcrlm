import pytest
from torch.nn import Linear
from source.nn.grad import AdamWithScheduling


@pytest.fixture(scope='module')
def temp_model():
    model = Linear(4,4)
    return model


@pytest.fixture(scope='function')
def adam_scheduling_decay(temp_model):
    optim = AdamWithScheduling(
        params=temp_model.parameters(),
        d_model=4,
        n_warmup_steps=5,
        lr_multiplier=2,
        scheduling=True,
        decay=True
    )
    return optim


@pytest.fixture(scope='function')
def adam_scheduling(temp_model):
    optim = AdamWithScheduling(
        params=temp_model.parameters(),
        d_model=4,
        n_warmup_steps=5,
        lr_multiplier=2,
        scheduling=True,
        decay=False
    )
    return optim


@pytest.fixture(scope='function')
def adam(temp_model):
    optim = AdamWithScheduling(
        params=temp_model.parameters(),
        d_model=4,
        n_warmup_steps=5,
        lr_multiplier=2,
        scheduling=False,
        decay=False
    )
    return optim


class TestScheduledAdamWithDecay:
    def test_step_num(self,adam_scheduling_decay):
        for i in range(1,1+5):
            assert adam_scheduling_decay.step_num == i
            adam_scheduling_decay.zero_grad()
            adam_scheduling_decay.step()


    def test_lr(self, adam_scheduling_decay):
        def calculate_lr(step_num):
            return 2 * (4 ** -0.5) * \
                min((step_num ** -0.5), (step_num * (5 ** -1.5)))
        
        for i in range(1,1+10):
            assert adam_scheduling_decay.lr == calculate_lr(i)
            adam_scheduling_decay.zero_grad()
            adam_scheduling_decay.step()
            
            for param_group in adam_scheduling_decay.optimiser.param_groups:
                assert param_group['lr'] == calculate_lr(i)
    

    def test_set_step_num(self, adam_scheduling_decay):
        with pytest.raises(AttributeError):
            adam_scheduling_decay.step_num = 10


    def test_set_lr(self, adam_scheduling_decay):
        with pytest.raises(AttributeError):
            adam_scheduling_decay.lr = 0.1


class TestScheduledAdam:
    def test_lr(self, adam_scheduling):
        def calculate_lr(step_num):
            return min(
                0.001,
                step_num / 5 * 0.001
            )
        
        for i in range(1,1+10):
            assert adam_scheduling.lr == calculate_lr(i)
            adam_scheduling.zero_grad()
            adam_scheduling.step()

            for param_group in adam_scheduling.optimiser.param_groups:
                assert param_group['lr'] == calculate_lr(i)


class TestAdam:
    def test_lr(self, adam):
        for _ in range(10):
            assert adam.lr == 0.001
            adam.zero_grad()
            adam.step()
            
            for param_group in adam.optimiser.param_groups:
                assert param_group['lr'] == 0.001