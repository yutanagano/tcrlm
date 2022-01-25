'''
data_handling.py
purpose: Python module with classes involved in the loading and preprocessing
         CDR3 data.
author: Yuta Nagano
ver: 2.5.1
'''


import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


amino_acids = {
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
}


tokens = (
    'A','C','D','E','F','G','H','I','K','L', # amino acids
    'M','N','P','Q','R','S','T','V','W','Y', # (0-19)
    '?', # mask token (20)
    '-'  # padding token (21)
)
token_to_index = dict()
index_to_token = dict()
for t, i in zip(tokens, range(len(tokens))):
    token_to_index[t] = i
    index_to_token[i] = t


def tokenise(cdr3: list) -> torch.Tensor:
    # Turn a cdr3 sequence from string form to tokenised tensor form.
    cdr3 = map(lambda x: token_to_index[x], cdr3)
    return torch.tensor(list(cdr3), dtype=torch.long)


def lookup(i: int) -> str:
    # Return the amino acid corresponding to the given token index.
    return index_to_token[i]


class CDR3Dataset(Dataset):
    # Custom dataset class to load CDR3 sequence data into memory and access it.
    def __init__(self,
                 path_to_csv: str,
                 p_mask: float = 0.15,
                 p_mask_random: float = 0.1,
                 p_mask_keep: float = 0.1,
                 jumble: bool = False):
        # Ensure that p_mask, p_mask_random and p_mask_keep values lie in a
        # well-defined range as probabilities
        assert(p_mask > 0 and p_mask < 1)
        assert(p_mask_random >= 0 and p_mask_random < 1)
        assert(p_mask_keep >= 0 and p_mask_keep < 1)
        assert(p_mask_random + p_mask_keep <= 1)

        super(CDR3Dataset, self).__init__()

        # Check that the specified csv exists, then load it as df
        if not (path_to_csv.endswith('.csv') and os.path.isfile(path_to_csv)):
            raise RuntimeError(f'Bad path to csv file: {path_to_csv}')
        dataframe = pd.read_csv(path_to_csv)

        # Save the dataframe as an attribute of the object
        self._dataframe = dataframe

        # Save the p_mask and related values as attributes of the object
        self._p_mask = p_mask
        self._p_random_threshold = p_mask_random
        self._p_keep_threshold = 1 - p_mask_keep

        # Enable/disable jumble mode
        self._jumble = jumble
    

    @property
    def jumble(self):
        return self._jumble
    

    @jumble.setter
    def jumble(self, b: bool):
        assert(type(b) == bool)
        self._jumble = b


    def __len__(self) -> int:
        # Return the length of the df as its own length
        return len(self._dataframe)


    def __getitem__(self, idx: int) -> (list, list):
        # Fetch the relevant cdr3 sequence from the dataframe
        cdr3 = self._dataframe.iloc[idx, 0]
        
        # If jumble mode is enabled, shuffle the sequence
        if self._jumble:
            cdr3 = list(cdr3)
            random.shuffle(cdr3)
            cdr3 = ''.join(cdr3)

        # Mask a proportion (p_mask) of the amino acids

        # 1) decide on which residues to mask
        i_to_mask = self._pick_masking_indices(len(cdr3))

        # 2) Based on the cdr3 sequence and the residue indices to mask,
        #    generate the input sequence
        x = self._generate_x(cdr3, i_to_mask)

        # 3) Based on the cdr3 sequence and the residue indices to mask,
        #    generate the target sequence
        y = self._generate_y(cdr3, i_to_mask)

        return (x, y)


    def _pick_masking_indices(self, cdr3_len: int) -> list:
        '''
        Given a particular length of cdr3, pick some residue indices at random
        to be masked.
        '''
        num_to_be_masked = max(1, round(cdr3_len * self._p_mask))
        return random.sample(range(cdr3_len), num_to_be_masked)
    

    def _generate_x(self, cdr3: str, indices: list) -> list:
        '''
        Given a cdr3 and a list of indices to be masked, generate an input
        sequence of tokens for model training, following the below convention:

        If an index i is chosen for masking, the residue at i is:
        - replaced with a random distinct token |self._p_mask_random of the time
        - kept as the original token            |self._p_mask_keep of the time
        - replaced with the mask token          |the rest of the time
        '''
        x = list(cdr3) # convert cdr3 str into a list of chars

        for i in indices: # for each residue to be replaced
            r = random.random() # generate a random float in range [0,1)

            if r < self._p_random_threshold: # opt.1:random distinct replacement
                x[i] = random.sample(tuple(amino_acids - {x[i]}),1)[0]

            elif r > self._p_keep_threshold: # opt.2:no replacement
                pass

            else: # opt.3:masking
                x[i] = '?'
        
        return x
    

    def _generate_y(self, cdr3: str, indices: list) -> list:
        '''
        Given a cdr3 and a list of indices to be masked, generate the target
        sequence, which will contain empty (padding) tokens for all indices
        except those that are masked in the input.
        '''
        y = ['-'] * len(cdr3)
        for i in indices: y[i] = cdr3[i]
        return y


class CDR3DataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int):
        super(CDR3DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
    

    def collate_fn(self, batch) -> (torch.Tensor, torch.Tensor):
        '''
        Helper collation function to be passed to the dataloader when loading
        batches from the CDR3Dataset.
        '''
        x_batch, y_batch = [], []
        for x_sample, y_sample in batch:
            x_batch.append(tokenise(x_sample))
            y_batch.append(tokenise(y_sample))

        x_batch = pad_sequence(
            sequences=x_batch,
            batch_first=True,
            padding_value=21
        )
        y_batch = pad_sequence(
            sequences=y_batch,
            batch_first=True,
            padding_value=21
        )

        return x_batch, y_batch