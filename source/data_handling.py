'''
data_handling.py
purpose: Python module with classes involved in the loading and preprocessing
         CDR3 data.
author: Yuta Nagano
ver: 2.2.0
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
        # Create and save token-to-index dictionaries for both input and output
        tokens_in = (
            '?', # mask token
            'A','C','D','E','F','G','H','I','K','L', # amino acids
            'M','N','P','Q','R','S','T','V','W','Y',
            '-' # padding token
        )
        self.token_dict_in = dict()
        for t, i in zip(tokens_in, range(len(tokens_in))):
            self.token_dict_in[t] = i

        tokens_out = (
            'A','C','D','E','F','G','H','I','K','L', # amino acids
            'M','N','P','Q','R','S','T','V','W','Y'
        )
        self.token_dict_out = dict()
        for t, i in zip(tokens_out, range(20)):
            self.token_dict_out[t] = i
        self.token_dict_out['-'] = 21 # add padding token at its correct index
    

    def tokenise_in(self, cdr3: str) -> torch.Tensor:
        '''
        Turn a cdr3 sequence from string form to tokenised tensor form (input
        version).
        '''
        cdr3 = map(lambda x: self.token_dict_in[x], cdr3)
        return torch.tensor(list(cdr3), dtype=torch.long)
    

    def tokenise_out(self, cdr3: str) -> torch.Tensor:
        '''
        Turn a cdr3 sequence from string form to tokenised tensor form (output
        version).
        '''
        cdr3 = map(lambda x: self.token_dict_out[x], cdr3)
        return torch.tensor(list(cdr3), dtype=torch.long)


class CDR3Dataset(Dataset):
    # Custom dataset class to load CDR3 sequence data into memory and access it.
    def __init__(self,
                 path_to_csv: str,
                 p_masked: float = 0.1):
        # Ensure that p_masked is in a well-defined range as a probability
        assert(p_masked > 0 and p_masked < 1)

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
        num_masked = max(1, int(num_residues * self.p_masked))
        i_to_mask = random.sample(range(num_residues), num_masked)

        # 2) mask those residues in the input
        cdr3_masked = list(cdr3) # convert cdr3 str into a list of chars
        for i in i_to_mask: cdr3_masked[i] = '?' # mask chars
        cdr3_masked = ''.join(cdr3_masked) # convert back to str

        # 3) hide/'pad' everything other than those residues in the target
        # Create a list of padding characters the same length as the cdr3
        cdr3_target = ['-'] * len(cdr3)
        for i in i_to_mask: cdr3_target[i] = cdr3[i]
        cdr3_target = ''.join(cdr3_target)

        return (cdr3_masked, cdr3_target)


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
    

    def collate_fn(self, batch: (str, str)) -> (torch.Tensor, torch.Tensor):
        '''
        Helper collation function to be passed to the dataloader when loading
        batches from the CDR3Dataset.
        '''
        x_batch, y_batch = [], []
        for x_sample, y_sample in batch:
            x_batch.append(self.tokeniser.tokenise_in(x_sample))
            y_batch.append(self.tokeniser.tokenise_out(y_sample))

        x_batch = pad_sequence(sequences=x_batch,
                               batch_first=True,
                               padding_value=21)
        y_batch = pad_sequence(sequences=y_batch,
                               batch_first=True,
                               padding_value=21)

        return x_batch, y_batch