import pytest
from source.datahandling import tokenisers
import torch


class TestAaTokeniser:
    @pytest.mark.parametrize(
        ('len_tuplet', 'expected_token_list'),
        (
            (1, [3, 2, 17, 17, 15, 21, 6]),
            (2, [22, 17, 317, 315, 281, 386]),
            (3, [417, 317, 6315, 6281, 5586])
        )
    )
    def test_tokenise(self, len_tuplet, expected_token_list):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=len_tuplet)

        aa_seq = 'CASSQYF'

        result = tokeniser.tokenise(aa_seq=aa_seq)
        expected = torch.tensor(expected_token_list, dtype=torch.long)

        assert result.equal(expected)


    @pytest.mark.parametrize(
        ('len_tuplet', 'tokenised_list'),
        (
            (1, [3,2,17,21,6]),
            (2, [22, 17, 321, 386]),
            (3, [417, 321, 6386])
        )
    )
    def test_generate_mlm_pair(self, len_tuplet, tokenised_list):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=len_tuplet)

        aa_seq = 'CASYF'
        tokenised = torch.tensor(tokenised_list, dtype=torch.long)

        result_x, result_y = tokeniser.generate_mlm_pair(
            aa_seq=aa_seq,
            p_mask=0.2,
            p_mask_random=0,
            p_mask_keep=0
        )

        def reconstruct(x, y):
            masked_indices = [i for i, t in enumerate(y) if t != 0]
            reconstructed = x.clone()
            for i in masked_indices:
                reconstructed[i] = y[i]
            return reconstructed, masked_indices

        reconstructed, masked_indices = reconstruct(result_x, result_y)

        assert type(result_x) == torch.Tensor
        assert type(result_y) == torch.Tensor
        assert len(masked_indices) in range(1,len_tuplet+1)
        assert reconstructed.equal(tokenised)


    @pytest.mark.parametrize(
        ('p_mask', 'expected_num_of_masked_i'),
        (
            (0.8, 4),
            (0.6, 3),
            (0.4, 2),
            (0.2, 1),
            (0, 0)
        )
    )
    def test_generate_mlm_pair_p_mask(self, p_mask, expected_num_of_masked_i):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        aa_seq = 'CASYF'

        _, result_y = tokeniser.generate_mlm_pair(
            aa_seq=aa_seq,
            p_mask=p_mask,
            p_mask_random=0,
            p_mask_keep=0
        )

        masked_indices = [i for i, t in enumerate(result_y) if t != 0]

        assert len(masked_indices) == expected_num_of_masked_i


    def test_generate_mlm_pair_mask_token(self):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        aa_seq = 'CASYF'

        result_x, result_y = tokeniser.generate_mlm_pair(
            aa_seq=aa_seq,
            p_mask=0.4,
            p_mask_random=0,
            p_mask_keep=0
        )

        masked_indices = [i for i, t in enumerate(result_y) if t != 0]

        assert all([result_x[i] == 1 for i in masked_indices])


    def test_generate_mlm_pair_random_token(self):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        aa_seq = 'CASYF'
        expected = [3,2,17,21,6]

        result_x, result_y = tokeniser.generate_mlm_pair(
            aa_seq=aa_seq,
            p_mask=0.4,
            p_mask_random=1,
            p_mask_keep=0
        )

        masked_indices = [i for i, t in enumerate(result_y) if t != 0]

        assert all(
            [
                result_x[i] in range(2,22) and \
                result_x[i] != expected[i] for i in masked_indices
            ]
        )


    def test_generate_mlm_pair_keep_token(self):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        aa_seq = 'CASYF'
        expected = [3,2,17,21,6]

        result_x, result_y = tokeniser.generate_mlm_pair(
            aa_seq=aa_seq,
            p_mask=0.4,
            p_mask_random=0,
            p_mask_keep=1
        )

        masked_indices = [i for i, t in enumerate(result_y) if t != 0]

        assert all([result_x[i] == expected[i] for i in masked_indices])


    @pytest.mark.parametrize(
        'len_tuplet', (-1, 'foo', None)
    )
    def test_error_bad_len_tuplet(self, len_tuplet):
        with pytest.raises(Exception):
            tokenisers.AaTokeniser(len_tuplet=len_tuplet)


    @pytest.mark.parametrize(
        'p_mask', (-0.1, 1)
    )
    def test_error_generate_mlm_pair_bad_p_mask(self, p_mask):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        with pytest.raises(RuntimeError):
            tokeniser.generate_mlm_pair(
                aa_seq='CASYF',
                p_mask=p_mask,
                p_mask_random=0,
                p_mask_keep=0
            )


    @pytest.mark.parametrize(
        'p_mask_random', (-0.1, 1.1)
    )
    def test_error_generate_mlm_pair_bad_p_mask_random(self, p_mask_random):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        with pytest.raises(RuntimeError):
            tokeniser.generate_mlm_pair(
                aa_seq='CASYF',
                p_mask=0.5,
                p_mask_random=p_mask_random,
                p_mask_keep=0
            )


    @pytest.mark.parametrize(
        'p_mask_keep', (-0.1, 1.1)
    )
    def test_error_generate_mlm_pair_bad_p_mask_keep(self, p_mask_keep):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        with pytest.raises(RuntimeError):
            tokeniser.generate_mlm_pair(
                aa_seq='CASYF',
                p_mask=0.5,
                p_mask_random=0,
                p_mask_keep=p_mask_keep
            )


    def test_error_generate_mlm_pair_bad_p_mask_random_keep(self):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=1)

        with pytest.raises(RuntimeError):
            tokeniser.generate_mlm_pair(
                aa_seq='CASYF',
                p_mask=0.5,
                p_mask_random=0.5,
                p_mask_keep=0.6
            )