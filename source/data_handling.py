'''
data_handling.py
purpose: Python module with classes involved in the loading and preprocessing
         CDR3 data.
author: Yuta Nagano
ver: 4.0.0
'''


import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


# Some useful data objects
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


# Helper functions
def tokenise(cdr3) -> torch.Tensor:
    '''
    Turn a cdr3 sequence from string form to tokenised tensor form.
    '''
    cdr3 = map(lambda x: token_to_index[x], cdr3)
    return torch.tensor(list(cdr3), dtype=torch.long)


def lookup(i: int) -> str:
    '''
    Return the amino acid corresponding to the given token index.
    '''
    return index_to_token[i]


def batch(data: list, batch_size: int) -> list:
    '''
    Take a list of items and segment it into smaller lists of size <= batch_size
    where all segments are of length batch_size until the very last batch which
    may be smaller if the length of the dataset is not exactly divisible.
    '''
    num_batches = (len(data) + batch_size - 1) // batch_size
    batched = []
    for i in range(num_batches):
        batched.append(
            data[
                i*batch_size:
                min((i+1)*batch_size,len(data))
            ]
        )
    return batched


def check_dataframe_format(dataframe: pd.DataFrame, columns: list) -> None:
    if dataframe.columns.tolist() != columns:
        raise RuntimeError(
            f'CSV file with incompatible format: columns '
            f'{dataframe.columns.tolist()}, expected {columns}.'
        )


# Dataset classes
class Cdr3PretrainDataset(Dataset):
    '''
    Custom dataset class to load unlabelled CDR3 sequence data into memory,
    access it, and perform preprocessing operations on it to pretrain an
    instance of Cdr3Bert on the data via a masked-amino acid modelling task.
    '''
    def __init__(
        self,
        path_to_csv: str,
        p_mask: float = 0.15,
        p_mask_random: float = 0.1,
        p_mask_keep: float = 0.1,
        jumble: bool = False
    ):
        # Ensure that p_mask, p_mask_random and p_mask_keep values lie in a
        # well-defined range as probabilities
        assert(p_mask > 0 and p_mask < 1)
        assert(p_mask_random >= 0 and p_mask_random < 1)
        assert(p_mask_keep >= 0 and p_mask_keep < 1)
        assert(p_mask_random + p_mask_keep <= 1)

        super(Cdr3PretrainDataset, self).__init__()

        # Check that the specified csv exists, then load it as df
        if not (path_to_csv.endswith('.csv') and os.path.isfile(path_to_csv)):
            raise RuntimeError(f'Bad path to csv file: {path_to_csv}')
        dataframe = pd.read_csv(path_to_csv)

        # Ensure that the input data is in the correct format
        check_dataframe_format(dataframe, ['CDR3', 'frequency'])

        # Save the dataframe as an attribute of the object
        self._dataframe = dataframe

        # Save a series containing the lengths of all CDR3s in the dataset
        self._cdr3_lens = dataframe['CDR3'].map(len)

        # Save the p_mask and related values as attributes of the object
        self._p_mask = p_mask
        self._p_random_threshold = p_mask_random
        self._p_keep_threshold = 1 - p_mask_keep

        # Enable/disable jumble mode
        self._jumble = jumble
    

    @property
    def jumble(self) -> bool:
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


    def get_length(self, idx: int) -> int:
        '''
        Return the length of the CDR3 sequence at the specified index
        '''
        return self._cdr3_lens.iloc[idx]


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


class Cdr3FineTuneDataset(Dataset):
    '''
    Custom dataset to load labelled CDR3 data (CDR3 and epitope pairs) into
    memory and access it for the fine-tuning phase of the Cdr3Bert network.
    '''
    def __init__(
        self,
        path_to_csv: str,
        p_matched_pair: float = 0.5
    ):
        # Ensure p_matched_pair takes on a well-defined value as a probability
        if not (p_matched_pair > 0 and p_matched_pair < 1):
            raise RuntimeError(
                f'Bad value for p_matched_pair: {p_matched_pair}. Value must be'
                'greater than 0 and less than 1.'
            )

        # Execute parent class initialisation
        super(Cdr3FineTuneDataset, self).__init__()

        # Check that the specified csv exists, then load it as df
        if not (path_to_csv.endswith('.csv') and os.path.isfile(path_to_csv)):
            raise RuntimeError(f'Bad path to csv file: {path_to_csv}')
        dataframe = pd.read_csv(path_to_csv)

        # Ensure that the input data is in the correct format
        check_dataframe_format(dataframe, ['Epitope', 'CDR3', 'Dataset'])

        # Save object attributes
        self._dataframe = dataframe
        self._epitope_groups = dataframe.groupby('Epitope')
        self._p_matched_pair = p_matched_pair


    def __len__(self) -> int:
        return len(self._dataframe)


    def __getitem__(self, idx: int) -> (str, str, int):
        # Fetch the relevant CDR3 from the dataframe.
        epitope_1, cdr3_1 = self._dataframe.iloc[idx,0:2]

        # Pick a second CDR3 sequence, which can either be an epitope-matched
        # CDR3 or a non-matched CDR3. How often matched or unmatched sequences
        # are picked as the second one will be influenced by the p_matched_pair
        # value passed to the dataset at creation. Along with picking the second
        # CDR3, we should also produce a label indicating whether the produced
        # pair is epitope-matched or not.
        epitope_2, cdr3_2, label = self._make_pair(idx)

        # Return pair with label.
        return cdr3_1, cdr3_2, label


    def _make_pair(self, idx: int) -> (str, str, int):
        '''
        Given the index to a reference epitope and CDR3, pick a second cdr3 to
        pair with the reference. The second CDR3 can be epitope-matched to the
        reference, or it can be unmatched. Which group to sample from to pick
        the second CDR3 will depend on the p_matched_pair value supplied to the
        constructor.
        '''
        r = random.random() # Generate a pseudorandom float in range [0, 1)

        # Based a psuedorandom float make a decision:
        if r < self._p_matched_pair:
            # Make a matched pair
            return self._get_matched_cdr3(idx)
        else:
            # Make an unmatched pair
            return self._get_unmatched_cdr3(idx)


    def _get_matched_cdr3(self, idx: int) -> (str, str, int):
        '''
        Given the index to a reference epitope and CDR3, pick a second epitope-
        matched CDR3 to pair with the reference.
        '''
        # Get the epitope of the reference
        ref_epitope = self._dataframe.iloc[idx, 0]

        # Get all members of the same epitope group except the reference itself
        matched_cdr3s = self._epitope_groups.get_group(ref_epitope).drop(idx)

        # Randomly sample one
        epitope_2, cdr3_2, _ = matched_cdr3s.sample().iloc[0]

        # Return the second CDR3 with its epitope, and a 'matched' label (1)
        return epitope_2, cdr3_2, 1


    def _get_unmatched_cdr3(self, idx: int) -> (str, str, int):
        '''
        Given the index to a reference epitope and CDR3, pick a second non-
        epitope-matched CDR3 to pair with the reference.
        '''
        # Get the epitope of the reference
        ref_epitope = self._dataframe.iloc[idx, 0]

        # Get all cdr3s not in the same epitope group
        unmatched_cdr3s = self._dataframe[
            self._dataframe['Epitope'] != ref_epitope
        ]

        # Randomly sample one
        epitope_2, cdr3_2, _ = unmatched_cdr3s.sample().iloc[0]

        # Return the second CDR3 with its epitope, and an 'unmatched' label (0)
        return epitope_2, cdr3_2, 0


# Sampler classes
class PadMinimalBatchSampler(Sampler):
    '''
    A custom batch sampler class designed to do almost-random batch sampling,
    but optimised to create batches of CDR3s with lengths that are relatively
    similar to each other. This is done to minimise the neccesity for padding
    (which increases the number of unnecessary computation) and therefore 
    reduce training time.
    '''
    def __init__(
        self,
        data_source: Cdr3PretrainDataset,
        batch_size: int,
        shuffle: bool = False
    ):
        assert(type(data_source) == Cdr3PretrainDataset)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data_source)

        self._len = (len(data_source) + batch_size - 1) // batch_size


    def __iter__(self):
        superbatched = batch(self._get_indices(), self.batch_size * 100)
        for sb in superbatched:
            sorted_sb = sorted(
                sb,
                key=lambda x: self.data_source.get_length(x)
            )
            batched = batch(sorted_sb, self.batch_size)
            if self.shuffle: random.shuffle(batched)
            for b in batched:
                yield b


    def __len__(self):
        return self._len


    def _get_indices(self):
        if self.shuffle:
            return random.sample(
                range(self.num_samples),
                k=self.num_samples
            )
        else:
            return range(self.num_samples)


# Dataloader classes
class Cdr3PretrainDataLoader(DataLoader):
    '''
    Custom dataloader class for feeding unlabelled CDR3 data to CDR3BERT during
    pretraining. With batch_optimisation on, it matches CDR3s that have
    relatively similar lengths to each other and puts them together in the same
    batch.
    '''
    def __init__(
        self,
        dataset: Cdr3PretrainDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
        distributed_sampler = None,
        batch_optimisation: bool = False
    ):
        assert(type(dataset) == Cdr3PretrainDataset)

        self._batch_optimisation = batch_optimisation

        if batch_optimisation:
            if distributed_sampler:
                raise RuntimeError(
                    'Cdr3PretrainDataLoader: distributed_sampler is mutually '\
                    'exclusive with batch_optimisation.'
                )
            super(Cdr3PretrainDataLoader, self).__init__(
                dataset=dataset,
                batch_sampler=PadMinimalBatchSampler(
                    data_source=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle
                ),
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        elif distributed_sampler:
            assert(type(distributed_sampler) == DistributedSampler)
            if shuffle:
                raise RuntimeError(
                    'Cdr3PretrainDataLoader: distributed_sampler is mutually '\
                    'exclusive with shuffle.'
                )
            super(Cdr3PretrainDataLoader, self).__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                sampler=distributed_sampler
            )
        else:
            super(Cdr3PretrainDataLoader, self).__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )


    @property
    def batch_optimisation(self):
        return self._batch_optimisation


    @property
    def jumble(self) -> bool:
        return self.dataset.jumble
    

    @jumble.setter
    def jumble(self, b: bool):
        self.dataset.jumble = b
    

    def collate_fn(self, batch) -> (torch.Tensor, torch.Tensor):
        '''
        Helper collation function to be passed to the dataloader when loading
        batches from the Cdr3PretrainDataset.
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


class Cdr3FineTuneDataLoader(DataLoader):
    '''
    Custom dataloader class for feeding labelled CDR3 data in the form of
    epitope-matched and unmatched pairs for the fine-tuning phase of CDR3BERT.
    '''
    def __init__(
        self,
        dataset: Cdr3FineTuneDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
        distributed_sampler = None
    ):
        assert(type(dataset) == Cdr3FineTuneDataset)

        if distributed_sampler:
            assert(type(distributed_sampler) == DistributedSampler)
            if shuffle:
                raise RuntimeError(
                    'Cdr3FineTuneDataLoader: distributed_sampler is mutually '\
                    'exclusive with shuffle.'
                )
            super(Cdr3FineTuneDataLoader, self).__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=distributed_sampler,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        else:
            super(Cdr3FineTuneDataLoader, self).__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
    

    def collate_fn(self, batch) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        '''
        Helper function which collates individual samples into tensor batches.
        '''
        x_1_batch, x_2_batch, y_batch = [], [], []
        for x_1_sample, x_2_sample, y_sample in batch:
            x_1_batch.append(tokenise(x_1_sample))
            x_2_batch.append(tokenise(x_2_sample))
            y_batch.append(y_sample)
        
        x_1_batch = pad_sequence(
            sequences=x_1_batch,
            batch_first=True,
            padding_value=21
        )
        x_2_batch = pad_sequence(
            sequences=x_2_batch,
            batch_first=True,
            padding_value=21
        )
        y_batch = torch.tensor(y_batch, dtype=torch.long)

        return x_1_batch, x_2_batch, y_batch