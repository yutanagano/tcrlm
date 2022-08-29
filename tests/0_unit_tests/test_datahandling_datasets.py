import pandas as pd
from pathlib import Path
import pytest
import random
import source.datahandling.datasets as datasets


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


def reconstruct(x, y):
    '''
    Using the x and y outputs of the pretrain dataset, reconstruct the original
    CDR3 sequence before masking.
    '''

    unmasked_in_y = [p for p in enumerate(y) if p[1] != '-']
    reconstructed = list(x)
    for idx, a_a in unmasked_in_y:
        assert(a_a in 'ACDEFGHIKLMNPQRSTVWY')
        reconstructed[idx] = a_a
    return ''.join(reconstructed)


class TestTcrDataset:
    def test_init_dataframe(self, unlabelled_data_df):
        dataset = datasets.TcrDataset(data=unlabelled_data_df)
        assert dataset._dataframe.equals(unlabelled_data_df)


    def test_init_path(self, path_to_unlabelled_data, unlabelled_data_df):
        dataset = datasets.TcrDataset(data=path_to_unlabelled_data)
        assert dataset._dataframe.equals(unlabelled_data_df)


    def test_init_nonexistent_csv(self):
        with pytest.raises(RuntimeError):
            datasets.TcrDataset(data='foobar/baz.csv')


    def test_init_noncsv_path(self):
        with pytest.raises(RuntimeError):
            datasets.TcrDataset(
                data='tests/0_unit_tests/test_datahandling_datasets.py'
            )


    def test_init_bad_type(self):
        with pytest.raises(RuntimeError):
            datasets.TcrDataset(data=1)


    def test_len(self, unlabelled_data_df):
        dataset = datasets.TcrDataset(data=unlabelled_data_df)
        assert len(dataset) == 29


class TestCdr3PretrainDataset:
    @pytest.mark.parametrize(
        ('p_mask'), (-0.1, 1.1)
    )
    def test_init_bad_p_mask(self, p_mask):
        with pytest.raises(AssertionError):
            datasets.Cdr3PretrainDataset(
                data=unlabelled_data_df,
                p_mask=p_mask
            )
    

    @pytest.mark.parametrize(
        ('p_mask_random'), (-0.1, 1.1)
    )
    def test_init_bad_p_mask_random(self, p_mask_random):
        with pytest.raises(AssertionError):
            datasets.Cdr3PretrainDataset(
                data=unlabelled_data_df,
                p_mask_random=p_mask_random
            )
    

    @pytest.mark.parametrize(
        ('p_mask_keep'), (-0.1, 1.1)
    )
    def test_init_bad_p_mask_keep(self, p_mask_keep):
        with pytest.raises(AssertionError):
            datasets.Cdr3PretrainDataset(
                data=unlabelled_data_df,
                p_mask_keep=p_mask_keep
            )
    

    def test_init_large_p_mask_random_keep_sum(self):
        with pytest.raises(AssertionError):
            datasets.Cdr3PretrainDataset(
                data=unlabelled_data_df,
                p_mask_random=0.6,
                p_mask_keep=0.6
            )


    def test_init_bad_format(self, path_to_bad_format_data):
        with pytest.raises(RuntimeError):
            datasets.Cdr3PretrainDataset(
                data=path_to_bad_format_data
            )
    

    def test_get_jumble(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            jumble=False
        )
        assert dataset._jumble == False
        assert dataset.jumble == False
    

    def test_set_jumble(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            jumble=False
        )
        dataset.jumble = True
        assert dataset._jumble == True


    def test_get_respect_frequencies(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            respect_frequencies=False
        )
        assert dataset._respect_frequencies == False
        assert dataset.respect_frequencies == False


    def test_set_respect_frequencies(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            respect_frequencies=False
        )
        dataset.respect_frequencies = True
        assert dataset._respect_frequencies == True


    @pytest.mark.parametrize(
        ('respect_frequencies', 'expected'),
        ((False, 29), (True, 34))
    )
    def test_len(self, unlabelled_data_df, respect_frequencies, expected):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            respect_frequencies=respect_frequencies
        )
        assert len(dataset) == expected


    @pytest.mark.parametrize(
        ('respect_frequencies', 'jumble', 'index', 'expected_reconstructed'),
        (
            (False, False, 0, 'CASRRREAFF'),
            (False, False, 28, 'CASSPTSRGPTPSGSYEQYF'),
            (True, False, 0, 'CASRRREAFF'),
            (True, False, 28, 'CASSGAGTSRNTQYF'),
            (True, False, 33, 'CASSPTSRGPTPSGSYEQYF'),
            (False, True, 28, 'CASSPTSRGPTPSGSYEQYF')
        )
    )
    def test_getitem(
        self,
        unlabelled_data_df,
        respect_frequencies,
        jumble,
        index,
        expected_reconstructed
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            respect_frequencies=respect_frequencies,
            jumble=jumble
        )
        result_x, result_y = dataset[index]
        reconstructed = reconstruct(result_x, result_y)

        assert type(result_x) == type(result_y) == list

        unmasked_in_y = [p for p in enumerate(result_y) if p[1] != '-']

        assert len(unmasked_in_y) in (2,3)

        if jumble:
            assert sorted(reconstructed) == sorted(expected_reconstructed)
            assert reconstructed != expected_reconstructed
            return
        
        assert reconstructed == expected_reconstructed


    @pytest.mark.parametrize(
        ('index', 'expected'),
        (
            (0, 'CASRRREAFF'),
            (28, 'CASSPTSRGPTPSGSYEQYF')
        )
    )
    def test_getitem_p_mask_zero(self, unlabelled_data_df, index, expected):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            p_mask=0
        )
        result_x, result_y = dataset[index]

        assert ''.join(result_x) == expected
        assert result_y == ['-'] * len(expected)


    def test_mask_token(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            p_mask=0.5,
            p_mask_random=0,
            p_mask_keep=0
        )

        for _ in range(10):
            idx = random.randrange(len(unlabelled_data_df))
            result_x, result_y = dataset[idx]

            unmasked_in_y = [p for p in enumerate(result_y) if p[1] != '-']
            
            assert all([result_x[i] == '?' for i, token in unmasked_in_y])
    

    def test_random_token(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            p_mask=0.5,
            p_mask_random=1,
            p_mask_keep=0
        )

        for _ in range(10):
            idx = random.randrange(len(unlabelled_data_df))
            expected = unlabelled_data_df.iloc[idx, 0]
            result_x, result_y = dataset[idx]

            unmasked_in_y = [p for p in enumerate(result_y) if p[1] != '-']
            
            assert all(
                [
                    result_x[i] in datasets.amino_acids and \
                    result_x[i] != expected[i] for i, token in unmasked_in_y
                ]
            )


    def test_keep_token(self, unlabelled_data_df):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            p_mask=0.5,
            p_mask_random=0,
            p_mask_keep=1
        )

        for _ in range(10):
            idx = random.randrange(len(unlabelled_data_df))
            expected = unlabelled_data_df.iloc[idx, 0]
            result_x, result_y = dataset[idx]

            unmasked_in_y = [p for p in enumerate(result_y) if p[1] != '-']
            
            assert all(
                [result_x[i] == expected[i] for i, token in unmasked_in_y]
            )


    @pytest.mark.parametrize(
        ('index'), (34, -35)
    )
    def test_dynamic_index_out_of_bounds(self, unlabelled_data_df, index):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            respect_frequencies=True
        )
        with pytest.raises(IndexError):
            dataset[index]


    @pytest.mark.parametrize(
        ('respect_frequencies', 'index', 'expected'),
        (
            (False, 28, 20),
            (True, 28, 15)
        )
    )
    def test_get_length(
        self,
        unlabelled_data_df,
        respect_frequencies,
        index,
        expected
    ):
        dataset = datasets.Cdr3PretrainDataset(
            data=unlabelled_data_df,
            respect_frequencies=respect_frequencies
        )
        result = dataset.get_length(index)
        assert result == expected


class TestCdr3FinetuneDataset:
    @pytest.mark.parametrize(
        ('p_matched_pair'),
        (0, 1)
    )
    def test_init_bad_p_matched_pair(self, labelled_data_df, p_matched_pair):
        with pytest.raises(RuntimeError):
            datasets.Cdr3FineTuneDataset(
                data=labelled_data_df,
                p_matched_pair=p_matched_pair
            )
    

    def test_init_bad_format(self, path_to_bad_format_data):
        with pytest.raises(RuntimeError):
            datasets.Cdr3FineTuneDataset(
                data=path_to_bad_format_data
            )


    def test_get_matched_cdr3(self, labelled_data_df):
        dataset = datasets.Cdr3FineTuneDataset(data=labelled_data_df)

        for _ in range(10):
            idx = random.randrange(len(labelled_data_df))

            ref_epitope = labelled_data_df.iloc[idx, 0]
            cdr3a, cdr3b, label = dataset._get_matched_cdr3(ref_epitope)

            assert label
            assert ref_epitope in \
                labelled_data_df[
                    (labelled_data_df['Alpha CDR3'] == cdr3a) &
                    (labelled_data_df['Beta CDR3'] == cdr3b)
                ]['Epitope'].unique()


    def test_get_unmatched_cdr3(self, labelled_data_df):
        dataset = datasets.Cdr3FineTuneDataset(data=labelled_data_df)
        
        for _ in range(10):
            idx = random.randrange(len(labelled_data_df))

            ref_epitope = labelled_data_df.iloc[idx, 0]
            cdr3a, cdr3b, label = dataset._get_unmatched_cdr3(ref_epitope)

            assert not label
            assert ref_epitope not in \
                labelled_data_df[
                    (labelled_data_df['Alpha CDR3'] == cdr3a) &
                    (labelled_data_df['Beta CDR3'] == cdr3b)
                ]['Epitope'].unique()


    def test_getitem(self, labelled_data_df):
        dataset = datasets.Cdr3FineTuneDataset(data=labelled_data_df)
        
        for _ in range(10):
            idx = random.randrange(len(labelled_data_df))

            cdr3_1a, cdr3_1b, cdr3_2a, cdr3_2b, label = dataset[idx]

            epitopes_1 = set(
                labelled_data_df[
                    (labelled_data_df['Alpha CDR3'] == cdr3_1a) &
                    (labelled_data_df['Beta CDR3'] == cdr3_1b)
                ]['Epitope'].unique()
            )

            epitopes_2 = set(
                labelled_data_df[
                    (labelled_data_df['Alpha CDR3'] == cdr3_2a) &
                    (labelled_data_df['Beta CDR3'] == cdr3_2b)
                ]['Epitope'].unique()
            )

            if label:
                assert not epitopes_1.isdisjoint(epitopes_2)
            else:
                assert epitopes_1.isdisjoint(epitopes_2)