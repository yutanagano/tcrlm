'Custom dataset classes.'


import random
import pandas as pd
from pathlib import Path
from source.datahandling import tokenisers
from source.utils.fileio import resolved_path_from_maybe_str
from source.utils.misc import check_dataframe_format
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, Union


class TcrDataset(Dataset):
    'Project custom base dataset class.'


    def __init__(
        self,
        data: Union[Path, str, pd.DataFrame],
        tokeniser: tokenisers.AaTokeniser
    ) -> None:
        super(TcrDataset, self).__init__()

        assert issubclass(type(tokeniser), tokenisers.AaTokeniser)
        self._tokeniser = tokeniser

        if type(data) == pd.DataFrame:
            self._dataframe = data
            return

        try:
            data = resolved_path_from_maybe_str(data)
        except(RuntimeError):
            raise RuntimeError(
                'data must be of type pd.DataFrame, Path, or str. '
                f'Got {type(data)}.'
            )

        if not (data.suffix == '.csv' and data.is_file()):
            raise RuntimeError(f'Bad path to csv file: {data}')
        
        self._dataframe = pd.read_csv(data)
    

    def __len__(self) -> int:
        return len(self._dataframe)


class Cdr3PretrainDataset(TcrDataset):
    'Dataset for masked-residue modelling.'


    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        tokeniser: tokenisers.AaTokeniser,
        p_mask: float = 0.15,
        p_mask_random: float = 0.1,
        p_mask_keep: float = 0.1,
        jumble: bool = False,
        respect_frequencies: bool = False
    ) -> None:
        super(Cdr3PretrainDataset, self).__init__(
            data=data,
            tokeniser=tokeniser
        )

        check_dataframe_format(self._dataframe, columns=['CDR3', 'frequency'])
        self._freq_cumsum = self._dataframe['frequency'].cumsum()

        self.p_mask = p_mask
        self.p_mask_random = p_mask_random
        self.p_mask_keep = p_mask_keep

        self.jumble = jumble
        self.respect_frequencies = respect_frequencies


    def __len__(self) -> int:
        if self.respect_frequencies:
            return self._freq_cumsum.iloc[-1]
        
        return len(self._dataframe)


    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Tensor, Tensor]:
        cdr3 = self._dataframe.iloc[self._dynamic_index(idx), 0]
        
        # If jumble mode is enabled, shuffle the sequence
        if self.jumble:
            cdr3 = list(cdr3)
            random.shuffle(cdr3)
            cdr3 = ''.join(cdr3)

        return self._tokeniser.generate_mlm_pair(
            aa_seq=cdr3,
            p_mask=self.p_mask,
            p_mask_random=self.p_mask_random,
            p_mask_keep=self.p_mask_keep
        )


    def get_length(self, idx: int) -> int:
        return len(self._dataframe.iloc[self._dynamic_index(idx), 0])


    def _dynamic_index(self, idx: int) -> str:
        '''
        If respect_frequencies is on, then transform idx with consideration of
        certain sequences (rows) appearing multiple times.
        '''
        if self.respect_frequencies:
            if idx >= self._freq_cumsum.iloc[-1] or \
                idx < -self._freq_cumsum.iloc[-1]:
                raise IndexError(
                    f'Index out of bounds: {idx} (len: {len(self)})'
                )
            
            idx %= self._freq_cumsum.iloc[-1]
            idx = self._freq_cumsum[self._freq_cumsum > idx].index[0]

        return idx


class Cdr3FineTuneDataset(TcrDataset):
    'Dataset for epitope-matching finetuning.'


    def __init__(
        self,
        data: str,
        tokeniser: tokenisers.AaTokeniser,
        p_matched_pair: float = 0.5
    ) -> None:
        if not (p_matched_pair >= 0 and p_matched_pair <= 1):
            raise RuntimeError(
                f'Bad value for p_matched_pair: {p_matched_pair}. Value must'
                'lie in [0, 1].'
            )

        super(Cdr3FineTuneDataset, self).__init__(
            data=data,
            tokeniser=tokeniser
        )

        check_dataframe_format(
            self._dataframe,
            columns=['Epitope', 'Alpha CDR3', 'Beta CDR3']
        )

        self._epitope_groups = self._dataframe.groupby('Epitope')
        self._p_matched_pair = p_matched_pair


    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        '''
        Fetch a pair of CDR3s, where the first CDR3 is the one located at the
        index specified, and the second is a randomly selected one, which is
        epitope-paired to the first with a probability of p_matched_pair.
        '''
        ref_epitope, cdr3_1a, cdr3_1b = self._dataframe.iloc[idx,0:3]
        cdr3_2a, cdr3_2b, label = self._make_pair(ref_epitope)

        return (
            self._tokeniser.tokenise(cdr3_1a),
            self._tokeniser.tokenise(cdr3_1b),
            self._tokeniser.tokenise(cdr3_2a),
            self._tokeniser.tokenise(cdr3_2b),
            label
        )


    def _make_pair(self, ref_epitope: str) -> Tuple[str, str, int]:
        '''
        Given a reference epitope, randomly select and return a TCR that is
        epitope-paired to the first with a probability of p_matched_pair.
        '''
        r = random.random()
        if r < self._p_matched_pair:
            return self._get_matched_cdr3(ref_epitope)
        else:
            return self._get_unmatched_cdr3(ref_epitope)


    def _get_matched_cdr3(self, ref_epitope: str) -> Tuple[str, str, int]:
        '''
        Given a reference epitope, randomly select and return an epitope-paired
        TCR.
        '''
        matched_cdr3s = self._epitope_groups.get_group(ref_epitope)
        _, cdr3_2a, cdr3_2b = matched_cdr3s.sample().iloc[0]

        # Return the second CDR3 with its epitope, and a 'matched' label (1)
        return cdr3_2a, cdr3_2b, 1


    def _get_unmatched_cdr3(self, ref_epitope: str) -> Tuple[str, str, int]:
        '''
        Given the index to a reference TCR, randomly select and return an
        epitope-paired TCR.
        '''
        unmatched_cdr3s = self._dataframe[
            self._dataframe['Epitope'] != ref_epitope
        ]
        _, cdr3_2a, cdr3_2b = unmatched_cdr3s.sample().iloc[0]

        # Return the second CDR3 with its epitope, and an 'unmatched' label (0)
        return cdr3_2a, cdr3_2b, 0