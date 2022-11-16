from src import optim
from torch.nn import Linear


class TestAdamWithScheduling:
    def test_lr(self):
        def calculate_lr(step_num):
            return 2 * (4 ** -0.5) * \
                min((step_num ** -0.5), (step_num * (5 ** -1.5)))

        dummy_model = Linear(3,3)
        adam = optim.AdamWithScheduling(
            params=dummy_model.parameters(),
            d_model=4,
            n_warmup_steps=5,
            lr_multiplier=2
        )
        
        for i in range(1,1+10):
            assert adam.lr == calculate_lr(i)
            adam.zero_grad()
            adam.step()
            
            for param_group in adam.optimiser.param_groups:
                assert param_group['lr'] == calculate_lr(i)