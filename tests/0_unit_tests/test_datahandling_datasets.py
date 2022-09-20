import pandas as pd
from pathlib import Path
import pytest
import random
from source.datahandling import datasets, tokenisers
import torch


@pytest.fixture(scope='module')
def path_to_bad_format_data():
    return Path('tests/resources/data/bad_format.csv')


@pytest.fixture(scope='module')
def path_to_unlabelled_data():
    return Path('tests/resources/data/mock_unlabelled.csv')


@pytest.fixture(scope='module')
def unlabelled_data_df(path_to_unlabelled_data):
    df = pd.read_csv(path_to_unlabelled_data)
    return df


@pytest.fixture(scope='module')
def labelled_data_df():
    df = pd.read_csv('tests/resources/data/mock_labelled.csv')
    return df


@pytest.fixture(scope='module')
def tokeniser():
    t = tokenisers.AaTokeniser(len_tuplet=1)
    return t


class TestTcrDataset:
    def test_init_dataframe(self, unlabelled_data_df, tokeniser):
        dataset = datasets.TcrDataset(
            data=unlabelled_data_df,
            tokeniser=tokeniser
        )
        assert dataset._dataframe.equals(unlabelled_data_df)


    def test_init_path(
        self,
        path_to_unlabelled_data,
        unlabelled_data_df,
        tokeniser
    ):
        dataset = datasets.TcrDataset(
            data=path_to_unlabelled_data,
            tokeniser=tokeniser
        )
        assert dataset._dataframe.equals(unlabelled_data_df)


    def test_error_nonexistent_csv(self, tokeniser):
        with pytest.raises(RuntimeError):
            datasets.TcrDataset(
                data='foobar/baz.csv',
                tokeniser=tokeniser
            )


    def test_error_noncsv_path(self, tokeniser):
        with pytest.raises(RuntimeError):
            datasets.TcrDataset(
                data='tests/0_unit_tests/test_datahandling_datasets.py',
                tokeniser=tokeniser
            )


    def test_error_bad_type(self, tokeniser):
        with pytest.raises(RuntimeError):
            datasets.TcrDataset(
                data=1,
                tokeniser=tokeniser
            )


    def test_len(self, unlabelled_data_df, tokeniser):
        dataset = datasets.TcrDataset(
            data=unlabelled_data_df,
            tokeniser=tokeniser
        )
        assert len(dataset) == 30


class TestCdr3PretrainDataset:
    @pytest.mark.parametrize(
        ('respect_frequencies', 'expected'),
        (
            (False, 30),
            (True, 35)
        )
    )
    def test_len(
        self,
        unlabelled_data_df,
        tokeniser,
        respect_frequencies,
        expected
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            tokeniser=tokeniser,
            respect_frequencies=respect_frequencies
        )

        assert len(dataset) == expected


    @pytest.mark.parametrize(
        ('respect_frequencies', 'jumble', 'index', 'token_list'),
        (
            (False, False, 1, [3,2,17,16,16,16,5,2,6,6]),
            (False, False, 29, [3,2,17,17,14,18,17,16,7,14,18,14,17,7,17,21,5,15,21,6]),
            (True, False, 1, [3,2,17,16,16,16,5,2,6,6]),
            (True, False, 29, [3,2,17,17,7,2,7,18,17,16,13,18,15,21,6]),
            (True, False, 34, [3,2,17,17,14,18,17,16,7,14,18,14,17,7,17,21,5,15,21,6]),
            (False, True, 29, [3,2,17,17,14,18,17,16,7,14,18,14,17,7,17,21,5,15,21,6])
        )
    )
    def test_getitem(
        self,
        unlabelled_data_df,
        tokeniser,
        respect_frequencies,
        jumble,
        index,
        token_list
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            tokeniser=tokeniser,
            respect_frequencies=respect_frequencies,
            jumble=jumble
        )

        result_x, result_y = dataset[index]
        expected = torch.tensor(token_list, dtype=torch.long)

        def reconstruct(x, y):
            masked_indices = [i for i, t in enumerate(y) if t != 0]
            reconstructed = x.clone()
            for i in masked_indices:
                reconstructed[i] = y[i]
            return reconstructed, masked_indices

        reconstructed, masked_indices = reconstruct(result_x, result_y)

        assert type(result_x) == torch.Tensor
        assert type(result_y) == torch.Tensor
        assert len(masked_indices) in (2, 3)

        if jumble:
            assert reconstructed.sort().values.equal(expected.sort().values)
            assert not reconstructed.equal(expected)
        else:
            assert reconstructed.equal(expected)


    @pytest.mark.parametrize(
        ('index'), (35, -36)
    )
    def test_dynamic_index_out_of_bounds(
        self,
        unlabelled_data_df,
        tokeniser,
        index
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            tokeniser=tokeniser,
            respect_frequencies=True
        )
        with pytest.raises(IndexError):
            dataset[index]


    @pytest.mark.parametrize(
        ('respect_frequencies', 'index', 'expected'),
        (
            (False, 29, 20),
            (True, 29, 15)
        )
    )
    def test_get_length(
        self,
        unlabelled_data_df,
        tokeniser,
        respect_frequencies,
        index,
        expected
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            tokeniser=tokeniser,
            respect_frequencies=respect_frequencies
        )
        result = dataset.get_length(index)
        assert result == expected


    def test_filter_for_tokeniser_tuplet_len(
        self,
        unlabelled_data_df
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            tokeniser=tokenisers.AaTokeniser(len_tuplet=3)
        )

        result_x, result_y = dataset[0]
        expected = torch.tensor([417, 316, 6296, 5896, 5885, 5662, 1206, 86], dtype=torch.long)

        def reconstruct(x, y):
            masked_indices = [i for i, t in enumerate(y) if t != 0]
            reconstructed = x.clone()
            for i in masked_indices:
                reconstructed[i] = y[i]
            return reconstructed, masked_indices

        reconstructed, _ = reconstruct(result_x, result_y)

        assert len(dataset) == 29
        assert reconstructed.equal(expected)


    def test_error_bad_format_data(self, path_to_bad_format_data, tokeniser):
        with pytest.raises(RuntimeError):
            datasets.Cdr3PretrainDataset(
                data=path_to_bad_format_data,
                tokeniser=tokeniser
            )


class TestCdr3FinetuneDataset:
    def test_getitem(self, labelled_data_df, tokeniser):
        dataset = datasets.Cdr3FineTuneDataset(
            data=labelled_data_df,
            tokeniser=tokeniser
        )
        
        for _ in range(10):
            idx = random.randrange(len(labelled_data_df))

            cdr3_1a, cdr3_1b, cdr3_2a, cdr3_2b, label = dataset[idx]

            epitopes_1 = self.get_epitope_set_of_tcr(
                cdr3a=cdr3_1a,
                cdr3b=cdr3_1b,
                df=labelled_data_df,
                tokeniser=tokeniser
            )

            epitopes_2 = self.get_epitope_set_of_tcr(
                cdr3a=cdr3_2a,
                cdr3b=cdr3_2b,
                df=labelled_data_df,
                tokeniser=tokeniser
            )

            if label:
                assert not epitopes_1.isdisjoint(epitopes_2)
            else:
                assert epitopes_1.isdisjoint(epitopes_2)


    @pytest.mark.parametrize(
        ('p_matched_pair', 'expected_label'),
        (
            (0, 0),
            (1, 1)
        )
    )
    def test_p_matched_pair(
        self,
        labelled_data_df,
        tokeniser,
        p_matched_pair,
        expected_label
    ):
        dataset = datasets.Cdr3FineTuneDataset(
            data=labelled_data_df,
            tokeniser=tokeniser,
            p_matched_pair=p_matched_pair
        )
        
        for _ in range(10):
            idx = random.randrange(len(labelled_data_df))

            cdr3_1a, cdr3_1b, cdr3_2a, cdr3_2b, label = dataset[idx]

            epitopes_1 = self.get_epitope_set_of_tcr(
                cdr3a=cdr3_1a,
                cdr3b=cdr3_1b,
                df=labelled_data_df,
                tokeniser=tokeniser
            )

            epitopes_2 = self.get_epitope_set_of_tcr(
                cdr3a=cdr3_2a,
                cdr3b=cdr3_2b,
                df=labelled_data_df,
                tokeniser=tokeniser
            )

            assert label == expected_label

            if expected_label:
                assert not epitopes_1.isdisjoint(epitopes_2)
            else:
                assert epitopes_1.isdisjoint(epitopes_2)


    @pytest.mark.parametrize(
        'p_matched_pair', (-0.1, 1.1)
    )
    def test_error_bad_p_matched_pair(
        self,
        labelled_data_df,
        tokeniser,
        p_matched_pair
    ):
        with pytest.raises(RuntimeError):
            datasets.Cdr3FineTuneDataset(
                data=labelled_data_df,
                tokeniser=tokeniser,
                p_matched_pair=p_matched_pair
            )
    

    def test_error_bad_format_data(self, path_to_bad_format_data, tokeniser):
        with pytest.raises(RuntimeError):
            datasets.Cdr3FineTuneDataset(
                data=path_to_bad_format_data,
                tokeniser=tokeniser
            )


    def get_epitope_set_of_tcr(self, cdr3a, cdr3b, df, tokeniser) -> set:
        cdr3a_tokenised = df['Alpha CDR3'].map(tokeniser.tokenise)
        cdr3b_tokenised = df['Beta CDR3'].map(tokeniser.tokenise)

        cdr3a_filter = cdr3a_tokenised.map(lambda x: x.equal(cdr3a))
        cdr3b_filter = cdr3b_tokenised.map(lambda x: x.equal(cdr3b))

        return set(
            df[cdr3a_filter & cdr3b_filter]['Epitope'].unique()
        )