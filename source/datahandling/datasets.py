'Custom dataset classes.'


import os
import random
import pandas as pd
from pathlib import Path
from source.utils.datahandling import amino_acids, check_dataframe_format
from source.utils.fileio import resolved_path_from_maybe_str
from torch.utils.data import Dataset
from typing import Tuple, Union


class TcrDataset(Dataset):
    'Project custom base dataset class.'

    def __init__(self, data: Union[Path, str, pd.DataFrame]) -> None:
        super(TcrDataset, self).__init__()

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
        respect_frequencies: bool = False,
        p_mask: float = 0.15,
        p_mask_random: float = 0.1,
        p_mask_keep: float = 0.1,
        jumble: bool = False
    ) -> None:
        assert (p_mask >= 0 and p_mask < 1)
        assert (p_mask_random >= 0 and p_mask_random <= 1)
        assert (p_mask_keep >= 0 and p_mask_keep <= 1)
        assert p_mask_random + p_mask_keep <= 1

        super(Cdr3PretrainDataset, self).__init__(data)

        check_dataframe_format(self._dataframe, columns=['CDR3', 'frequency'])
        self._freq_cumsum = self._dataframe['frequency'].cumsum()

        self._p_mask = p_mask
        self._p_random_threshold = p_mask_random
        self._p_keep_threshold = 1 - p_mask_keep

        self.jumble = jumble
        self.respect_frequencies = respect_frequencies
    

    @property
    def jumble(self) -> bool:
        return self._jumble
    

    @jumble.setter
    def jumble(self, b: bool):
        assert(type(b) == bool)
        self._jumble = b

    
    @property
    def respect_frequencies(self) -> bool:
        return self._respect_frequencies

    
    @respect_frequencies.setter
    def respect_frequencies(self, b: bool):
        assert(type(b) == bool)
        self._respect_frequencies = b


    def __len__(self) -> int:
        if self._respect_frequencies:
            return self._freq_cumsum.iloc[-1]
        
        return len(self._dataframe)


    def _dynamic_index(self, idx: int) -> str:
        '''
        If respect_frequencies is on, then transform idx with consideration of
        certain sequences (rows) appearing multiple times.
        '''
        if self._respect_frequencies:
            if idx >= self._freq_cumsum.iloc[-1] or \
                idx < -self._freq_cumsum.iloc[-1]:
                raise IndexError(
                    f'Index out of bounds: {idx} (len: {len(self)})'
                )
            
            idx %= self._freq_cumsum.iloc[-1]
            idx = self._freq_cumsum[self._freq_cumsum > idx].index[0]

        return idx


    def __getitem__(
        self,
        idx: int
    ) -> Union[Tuple[list, list], Tuple[str, str]]:
        cdr3 = self._dataframe.iloc[self._dynamic_index(idx), 0]
        
        # If jumble mode is enabled, shuffle the sequence
        if self._jumble:
            cdr3 = list(cdr3)
            random.shuffle(cdr3)
            cdr3 = ''.join(cdr3)

        i_to_mask = self._pick_masking_indices(len(cdr3))
        masked_cdr3 = self._generate_masked(cdr3, i_to_mask)
        target = self._generate_target(cdr3, i_to_mask)

        return (masked_cdr3, target)


    def get_length(self, idx: int) -> int:
        return len(self._dataframe.iloc[self._dynamic_index(idx), 0])


    def _pick_masking_indices(self, cdr3_len: int) -> list:
        if self._p_mask == 0:
            return []
        
        num_to_be_masked = max(1, round(cdr3_len * self._p_mask))
        return random.sample(range(cdr3_len), num_to_be_masked)
    

    def _generate_masked(self, cdr3: str, indices: list) -> list:
        '''
        Given a cdr3 and a list of indices to be masked, generate an input
        sequence of tokens for model training, following the below convention:

        If an index i is chosen for masking, the residue at i is:
        - replaced with a random distinct token|self._p_mask_random of the time
        - kept as the original token           |self._p_mask_keep of the time
        - replaced with the mask token         |the rest of the time
        '''
        cdr3 = list(cdr3)

        for i in indices:
            r = random.random()
            if r < self._p_random_threshold:
                # Ensure residue is replaced with a DISTINCT residue
                cdr3[i] = random.sample(tuple(amino_acids - {cdr3[i]}),1)[0]
                continue
            if r < self._p_keep_threshold:
                cdr3[i] = '?'
                continue
        
        return cdr3
    

    def _generate_target(self, cdr3: str, indices: list) -> list:
        '''
        Given a cdr3 and a list of indices to be masked, generate the target
        sequence, which will contain empty (padding) tokens for all indices
        except those that are masked in the input.
        '''
        target = ['-'] * len(cdr3)

        for i in indices:
            target[i] = cdr3[i]
        
        return target


class Cdr3FineTuneDataset(TcrDataset):
    'Dataset for epitope-matching finetuning.'

    def __init__(self, data: str, p_matched_pair: float = 0.5) -> None:
        if not (p_matched_pair > 0 and p_matched_pair < 1):
            raise RuntimeError(
                f'Bad value for p_matched_pair: {p_matched_pair}. Value must'
                'be greater than 0 and less than 1.'
            )

        super(Cdr3FineTuneDataset, self).__init__(data)

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

        return cdr3_1a, cdr3_1b, cdr3_2a, cdr3_2b, label


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