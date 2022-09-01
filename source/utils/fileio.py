'Housekeeping utilities involving saving and loading files.'


import pandas as pd
from pathlib import Path
from shutil import rmtree
from source.nn import models
from source.utils.misc import check_dataframe_format, print_with_deviceid
import torch
from typing import Union


def resolved_path_from_maybe_str(path: Union[Path, str]) -> Path:
    if issubclass(type(path), Path):
        return path.resolve()
    
    if type(path) == str:
        return Path(path).resolve()

    raise RuntimeError(f'Unknown type for path: {type(path)}')


def create_training_run_directory(
    working_directory: Union[Path, str],
    run_id: str,
    mode: str,
    overwrite: bool = False
) -> Path:
    '''
    Creates a directory for a training run with the specified run ID. The
    directory will be created in a 'pretrain_runs' parent directory if mode
    is set to 'pretrain', and in the 'finetune_runs' parent directory if 
    mode is set to 'finetune'. If the relevant parent directory does not
    yet exist in the working directory, then it will bre created first, and
    then the training run directory will be created inside it. 
    
    The function returns a string path to the newly created training run
    directory.
    
    If 'overwrite' is specified and a training run directory of the
    same name/run ID already exists, it and its contents will be deleted/
    overwritten. Otherwise, the existence of a similarly named directory 
    will cause the program to systematically search for an alternative and
    available run ID.
    '''

    working_directory = resolved_path_from_maybe_str(working_directory)
    if not working_directory.is_dir():
        raise RuntimeError('The specified working directory does not exist.')

    if mode in {'pretrain', 'p'}:
        parent_dir = working_directory / 'pretrain_runs'
    elif mode in {'finetune', 'f'}:
        parent_dir = working_directory / 'finetune_runs'
    else:
        raise RuntimeError(f'Unknown mode: {mode}')

    if not parent_dir.is_dir():
        print(f'Creating new parent directory {parent_dir.name}...')
        parent_dir.mkdir()

    training_run_dir = parent_dir / run_id

    if training_run_dir.is_dir():
        if overwrite:
            rmtree(training_run_dir)
        else:
            # Suffix an incrementing integer until available directory name
            print(
                f'The directory {training_run_dir} already exists. '
                'Searching for an alternative...'
            )

            suffix_int = 0
            while training_run_dir.is_dir():
                suffix_int += 1
                training_run_dir = parent_dir / f'{run_id}_{suffix_int}'

            print(f'Run directory adjusted to {training_run_dir}.')

    training_run_dir.mkdir()
    
    return training_run_dir


class TrainingRecordManager:
    'Manager for record keeping during training.'

    def __init__(
        self,
        training_run_dir: Union[Path, str],
        distributed: bool,
        device: torch.device,
        test_mode: bool = False
    ) -> None:
        training_run_dir = resolved_path_from_maybe_str(training_run_dir)
        if not training_run_dir.is_dir():
            raise RuntimeError(
                'The specified training run directory does not exist.')
        
        self._training_run_dir = training_run_dir
        self._distributed = distributed
        self._device = device
        self._test_mode = test_mode


    def save_log(self, log_dict: dict) -> None:
        '''
        Saves the given training stats log as a csv inside the training run
        directory. The specific way in which the saved file is named is
        determined from the distributed variable.
        '''

        if self._distributed:
            destination = self._training_run_dir / \
                f'training_log_{self._device}.csv'.replace(':','_') # colons are illegal in windows filenames so 'cuda:0' is not allowed
        else:
            destination = self._training_run_dir / f'training_log.csv'
        
        print_with_deviceid(
            f'Saving training log to {destination}...',
            self._device
        )

        log_df = pd.DataFrame.from_dict(data=log_dict, orient='index')
        log_df.to_csv(destination, index_label='epoch')


    def _decompose_state_dicts(
        self,
        model: torch.nn.Module
    ) -> dict[str, dict]:
        if type(model) == models.Cdr3BertPretrainWrapper:
            return {
                'bert': model.bert.state_dict(),
                'generator': model.generator.state_dict()
            }
        
        if type(model) == models.Cdr3BertFineTuneWrapper:
            return {
                'alpha_bert': model.embedder.alpha_bert.state_dict(),
                'beta_bert': model.embedder.beta_bert.state_dict(),
                'classifier': model.classifier.state_dict()
            }


    def save_model(self, model: torch.nn.Module) -> None:
        '''
        If appropriate, save the given model inside the given directory.
        Whether it is appropriate for the current process to save a copy of the
        model is determined based on the distributed, device and test_mode
        variables.
        '''

        if self._distributed and \
            self._device.index != 0 and \
            not self._test_mode:
            return

        if self._distributed:
            model = model.module

        state_dicts = self._decompose_state_dicts(model)

        assert len(state_dicts) > 0
        
        for module_name in state_dicts:
            state_dict = state_dicts[module_name]

            filename = f'{module_name}_state_dict'
            if self._test_mode:
                filename += f'_{self._device}'.replace(':','_')
            destination = self._training_run_dir/(filename+'.pt')

            print_with_deviceid(
                f'Saving pretrained model to {destination}...',
                self._device
            )
            torch.save(state_dict, destination)


# Hyperparameter parsing/loading
def _bool_convert(x: str):
    '''
    Converts string to bool values, where 'True' is mapped to True, and 'False'
    is mapped to False. Helper function to hyperparams.
    '''

    if x == 'True':
        return True

    if x == 'False':
        return False
    
    raise RuntimeError(f'Expected "True" or "False", got {x}')


def parse_hyperparams(csv_path: Union[Path, str]) -> dict:
    '''
    Read a csv file and extract a list of hyperparameters from it. Then save
    that information in the form of a python dictionary. The csv should have
    a header as follows: "param_name,type,value". The first column should
    have the name of the hyperparameter (which will become the hyperparameter's
    dictionary key), the second column should have the data type that the value
    should be converted to (e.g. int, str, etc. See below for a list of
    supported types.), and the third column should contain the value.

    Supported types:
    Type        Alias (to put in the csv)
    -------------------------------------
    Boolean     bool
    Float       float
    Integer     int
    String      str
    '''

    csv_path = resolved_path_from_maybe_str(csv_path)

    if not (csv_path.suffix == '.csv' and csv_path.is_file()):
        raise RuntimeError(f'Bad path to csv: {csv_path}.')

    df = pd.read_csv(csv_path)
    check_dataframe_format(df, ['param_name', 'type', 'value'])

    type_dict = {
        'bool': _bool_convert,
        'float': float,
        'int': int,
        'str': str
    }

    hyperparams = dict()
    for _, row in df.iterrows():
        if not row['type'] in type_dict:
            raise RuntimeError(f'Unrecognised data type: {row["type"]}')
        
        param_value = type_dict[row['type']](row['value'])
        hyperparams[row['param_name']] = param_value

    return hyperparams