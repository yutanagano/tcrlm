from src.modules.bert.embedding.sinpos import *
import torch


class TestSinPositionEmbedding:
    def test_embedding(self):
        embedding = SinPositionEmbedding(num_embeddings=2, embedding_dim=4)
        result = embedding(torch.tensor([[1,2],[1,0]], dtype=torch.long))
        expected = torch.tensor(
            [
                [[0.0000,1.0000,0.0000,1.0000],[0.8415,0.5403,0.1816,0.9834]],
                [[0.0000,1.0000,0.0000,1.0000],[0.0000,0.0000,0.0000,0.0000]]
            ]
        )

        torch.testing.assert_close(result, expected, rtol=0.001, atol=0)


class TestSinPositionEmbeddingRelative:
    def test_embedding(self):
        embedding = SinPositionEmbeddingRelative(embedding_dim=4, sin_scale_factor=2)
        result = embedding(torch.tensor([[[1,3],[2,3],[3,3]],[[1,2],[2,2],[0,0]]], dtype=torch.float32))
        expected = torch.tensor(
            [
                [
                    [0.0000,1.0000,0.0000,1.0000],
                    [0.0000,-1.0000,0.0000,1.0000],
                    [0.0000,1.0000,0.0000,1.0000],
                ],
                [
                    [0.0000,1.0000,0.0000,1.0000],
                    [0.0000,1.0000,0.0000,1.0000],
                    [0.0000,0.0000,0.0000,0.0000],
                ]
            ]
        )

        torch.testing.assert_close(result, expected)