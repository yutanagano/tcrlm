'''
cdr3dataset.py
purpose: Python module with classes involved in the loading and preprocessing
         CDR3 data.
author: Yuta Nagano
ver: 2.0.0
'''


import os
import random
import pandas as pd
from torch.utils.data import Dataset


class CDR3Dataset(Dataset):
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