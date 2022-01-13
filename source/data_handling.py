'''
data_handling.py
purpose: Python module with classes involved in the loading and preprocessing
         CDR3 data.
author: Yuta Nagano
ver: 2.1.0
'''


import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CDR3Tokeniser:
    '''
    Helper class that can translate CDR3 sequences between their string
    representations and their tokenised tensor representations.
    '''
    def __init__(self):
        # Create and save token-to-index and index-to-token dictionaries
        tokens = (
            '?', # mask token
            'A','C','D','E','F','G','H','I','K','L', # amino acids
            'M','N','P','Q','R','S','T','V','W','Y',
            '-'  # padding token
        )
        self.token_to_index_dict = dict()
        self.index_to_token_dict = dict()
        for t, i in zip(tokens, range(len(tokens))):
            self.token_to_index_dict[t] = i
            self.index_to_token_dict[i] = t
    

    def tokenise(self, cdr3: str) -> torch.Tensor:
        # Turn a cdr3 sequence from string form to tokenised tensor form
        cdr3 = map(lambda x: self.token_to_index_dict[x], cdr3)
        return torch.tensor(list(cdr3), dtype=torch.int)


    def to_string(self, tokenised_cdr3: torch.Tensor) -> str:
        # Turn a cdr3 sequence from tokenised tensor form to string form
        return ''.join(map(lambda x: self.index_to_token_dict[x.item()],
                           tokenised_cdr3))


class CDR3Dataset(Dataset):
    # Custom dataset class to load CDR3 sequence data into memory and access it.
    def __init__(self,
                 path_to_csv: str,
                 p_masked: float = 0):
        # Super init
        super(CDR3Dataset, self).__init__()

        # Check that the specified csv exists, then load it as df
        if not (path_to_csv.endswith('.csv') and os.path.isfile(path_to_csv)):
            raise RuntimeError(f'Bad path to csv file: {path_to_csv}')
        dataframe = pd.read_csv(path_to_csv)

        # Save the dataframe as an attribute of the object
        self.dataframe = dataframe

        # Save the p_masked value as an attribute of the object
        self.p_masked = p_masked


    def __len__(self) -> int:
        # Return the length of the df as its own length
        return len(self.dataframe)


    def __getitem__(self, idx: int) -> (str, str):
        # Fetch the relevant cdr3 sequence from the dataframe
        cdr3 = self.dataframe.iloc[idx, 0]

        # Mask a proportion (p_masked) of the amino acids
        # 1) decide on which residues to mask
        num_residues = len(cdr3)
        num_masked = 0
        if self.p_masked:
            num_masked = max(1, int(num_residues * self.p_masked))
        i_to_mask = random.sample(range(num_residues), num_masked)
        # 2) mask those residues
        cdr3_masked = list(cdr3) # convert cdr3 str into a list of chars
        for i in i_to_mask: cdr3_masked[i] = '?' # mask chars
        cdr3_masked = ''.join(cdr3_masked) # convert back to str

        return (cdr3_masked, cdr3)


class CDR3DataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int):
        super(CDR3DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        # Create and save an instance of a CDR3Tokeniser
        self.tokeniser = CDR3Tokeniser()
    

    def collate_fn(self, batch):
        '''
        Helper collation function to be passed to the dataloader when loading
        batches from the CDR3Dataset.
        '''
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.tokeniser.tokenise(src_sample))
            tgt_batch.append(self.tokeniser.tokenise(tgt_sample))

        src_batch = pad_sequence(sequences=src_batch,
                                 batch_first=True,
                                 padding_value=21)
        tgt_batch = pad_sequence(sequences=tgt_batch,
                                 batch_first=True,
                                 padding_value=21)

        return src_batch, tgt_batch