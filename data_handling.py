'''
data_handling.py
purpose: Python module with classes involved in the loading and preprocessing
         CDR3 data.
author: Yuta Nagano
ver: 1.0.4
'''


import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

amino_acids = (
    'A',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'K',
    'L',
    'M',
    'N',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'V',
    'W',
    'Y'
)


class SequenceConverter():
    def __init__(self, padding: int, norm_atchley: bool):
        # Create dictionaries mapping amino acids to their atchley factor
        # encodings, and one-hot encodings respectively

        # Create two dicts with just null entries (used later for zero-padding if necessary)
        self.atchley_dict = {'NULL': torch.zeros(5,dtype=torch.float32)}
        self.one_hot_dict = {'NULL': torch.zeros(20,dtype=torch.float32)}

        # Load atchley factor data, stored in a csv
        path_to_atchley_csv = '/home/yuta/Projects/cdr3encoding/atchley_factors.csv'
        atchley_table = pd.read_csv(path_to_atchley_csv,index_col=0)

        if norm_atchley:
            for col in atchley_table.columns:
                atchley_table[col] = atchley_table[col] - atchley_table[col].min()
                atchley_table[col] = atchley_table[col] / atchley_table[col].max() * 2 - 1

        # Prepare a 20x20 identity matrix (useful for one-hot encodings)
        i_matrix = torch.eye(20,dtype=torch.float32)

        # Now loop through all amino acids and populate both dictionaries with
        # their corresponding encodings
        for i, letter in enumerate(amino_acids):
            # Atchley factors
            self.atchley_dict[letter] = torch.from_numpy(atchley_table.loc[letter].to_numpy(dtype=np.float32))

            # One-hot encodings
            self.one_hot_dict[letter] = i_matrix[i]
        
        # Save padding settings (Padding can be set to any non-negative integer.
        # If padding=x>0, input sequences are zero-padded to length x. If
        # padding=0, no padding is performed. Given padding=x>0, any input
        # sequences of length>x will throw an error.)
        self.padding = padding


    def _check_query(self, aa: str) -> None:
        # Ensure query is a string
        if type(aa) != str: raise RuntimeError(f'Query must be a string: (not {type(aa)}).')
        # Ensure that the input sequence length is compatible with padding setting
        if self.padding and len(aa) > self.padding:
            raise RuntimeError(f'Input sequence {aa} is too long (padding={self.padding}).')


    def _check_letter(self,letter: str) -> None:
        if not letter in amino_acids:
            raise RuntimeError(f'Factor for amino acid "{letter}" not found.')


    def _produce_encoding(self, aa: str, atchley: bool) -> list:
        # Ensure that query is valid
        self._check_query(aa)

        # Fetch the relevant dictionary
        if atchley: dct = self.atchley_dict
        else: dct = self.one_hot_dict

        # Create an array to store the individual amino acid encodings
        factors = []
        
        # Loop through each letter in the query sequence and append to list
        for letter in aa:
            # Ensure that the letter is a valid amino acid
            self._check_letter(letter)
            
            # Add the factor for the amino acid to the list
            factors.append(dct[letter])
        
        for i in range(self.padding - len(aa)):
            factors.append(dct['NULL'])

        return torch.t(torch.stack(factors))


    def to_atchley(self, aa: str) -> torch.Tensor:
        return self._produce_encoding(aa, True)


    def to_one_hot(self, aa: str) -> torch.Tensor:
        return self._produce_encoding(aa, False)


class CDR3Dataset(Dataset):
    def __init__(self,
                 path_to_csv: str,
                 x_atchley: bool = True,
                 y_atchley: bool = False,
                 padding: int = 0,
                 norm_atchley: bool = True):
        # Super init
        super(CDR3Dataset, self).__init__()

        # Check that the specified csv exists, then load it as df
        if not (path_to_csv.endswith('.csv') and os.path.isfile(path_to_csv)):
            raise RuntimeError(f'Bad path to csv file: {path_to_csv}')
        self.dataframe = pd.read_csv(path_to_csv)

        # Save x_atchley and y_atchley values
        self.x_atchley = x_atchley
        self.y_atchley = y_atchley

        # Create an instance of the atchley converter
        self.converter = SequenceConverter(padding=padding,norm_atchley=norm_atchley)


    def __len__(self) -> int:
        # Return the length of the df as its own length
        return len(self.dataframe)


    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # Fetch the relevant cdr3 sequence from the dataframe
        cdr3 = self.dataframe.iloc[idx, 0]

        # If both x and y are of the same form, we can save computation
        if self.x_atchley == self.y_atchley:
            # If both are atchley form
            if self.x_atchley:
                a_encoding = self.converter.to_atchley(cdr3)
                return a_encoding, a_encoding
            
            # If both are one_hot form
            oh_encoding = self.converter.to_one_hot(cdr3)
            return oh_encoding, oh_encoding

        # Otherwise, compute both
        a_encoding = self.converter.to_atchley(cdr3)
        oh_encoding = self.converter.to_one_hot(cdr3)

        if self.x_atchley:
            x = a_encoding
            y = oh_encoding
        else:
            x = oh_encoding
            y = a_encoding

        return x, y
    

    def get_cdr3(self, idx: int) -> str:
        # Fetch the string representing the CDR3 at the given index, return it
        return self.dataframe.iloc[idx, 0]