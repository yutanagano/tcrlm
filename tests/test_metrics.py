from src import metrics
import torch


class TestAlignment:
    def test_alignment(self):
        x = torch.tensor(
            [
                [ 0, 1],
                [ 1, 0],
                [ 0,-1],
                [-1, 0]
            ],
            dtype=torch.float32
        )
        labels = torch.tensor([0,0,1,1])

        torch.testing.assert_close(
            metrics.alignment(x,labels,alpha=2),
            torch.tensor(2, dtype=torch.float32)
        )


class TestUniformity:
    def test_uniformity(self):
        x = torch.tensor(
            [
                [ 0, 1],
                [ 1, 0],
                [ 0,-1],
                [-1, 0]
            ],
            dtype=torch.float32
        )

        torch.testing.assert_close(
            metrics.uniformity(x,t=2),
            torch.log(torch.tensor(1/3)) + \
                torch.log(
                    torch.tensor(2.0) + torch.exp(torch.tensor(-4.0))
                ) - \
                torch.tensor(4.0)
        )