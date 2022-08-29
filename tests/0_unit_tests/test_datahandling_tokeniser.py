import pytest
from source.datahandling import tokenisers
import torch


class TestAaTokeniser:
    @pytest.mark.parametrize(
        'len_tuplet', (-1, 'foo', None)
    )
    def test_error_if_bad_len_tuplet(self, len_tuplet):
        with pytest.raises(Exception):
            tokenisers.AaTokeniser(len_tuplet=len_tuplet)


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
        ('len_tuplet', 'expected_token_list'),
        (
            (1, [3, 2, 17, 1, 15, 21, 6]),
            (2, [22, 17, 1, 1, 281, 386]),
            (3, [417, 1, 1, 1, 5586])
        )
    )
    def test_tokenise_masked(self, len_tuplet, expected_token_list):
        tokeniser = tokenisers.AaTokeniser(len_tuplet=len_tuplet)

        aa_seq = 'CAS?QYF'

        result = tokeniser.tokenise(aa_seq=aa_seq)
        expected = torch.tensor(expected_token_list, dtype=torch.long)

        assert result.equal(expected)