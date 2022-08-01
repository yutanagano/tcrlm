'Housekeeping utilities involving saving and loading files.'


import os
import pandas as pd
import shutil
from source.utils.datahandling import check_dataframe_format
from source.utils.misc import print_with_deviceid
import torch


def create_training_run_directory(
    run_id: str,
    mode: str,
    overwrite: bool = False
) -> str:
    '''
    Creates a directory for a training run with the specified run ID. The
    directory will be created in the 'pretrain_runs' parent directory if mode
    is set to 'pretrain', and in the 'finetune_runs' parent directory if mode
    is set to 'finetune'. If the relevant parent directory does not yet exist
    in the working directory, then it will bre created first, and then the
    training run directory will be created inside it. 
    
    The function returns a string path to the newly created training run
    directory.
    
    If 'overwrite' is specified and a training run directory of the
    same name/run ID already exists, it and its contents will be deleted/
    overwritten. Otherwise, the existence of a similarly named directory will
    cause the program to systematically search for an alternative and available
    run ID.
    '''

    if mode in {'pretrain', 'p'}:
        parent_dir = 'pretrain_runs'
    elif mode in {'finetune', 'f'}:
        parent_dir = 'finetune_runs'
    else:
        raise RuntimeError(f'Unknown mode: {mode}')

    # Create parent directory if not already existent
    if not os.path.isdir(parent_dir):
        print(f'Creating new parent directory {parent_dir}...')
        os.mkdir(parent_dir)

    # Create a path to a target directory corresponding to the specified run_id
    tr_dir = os.path.join(parent_dir,run_id)

    # If there already exists a directory at the specified path/name
    if os.path.isdir(tr_dir):
        # If overwrite=True, delete the preexisting directory
        if overwrite: shutil.rmtree(tr_dir)
        # Otherwise, keep modifying the target directory path in a systematic
        # way until we find a directory path that does not yet exist.
        else:
            suffix_int = 1
            new_tr_dir = f'{tr_dir}_{suffix_int}'
            while os.path.isdir(new_tr_dir):
                suffix_int += 1
                new_tr_dir = f'{tr_dir}_{suffix_int}'
            # Quick user feedback
            print(
                f'A directory {tr_dir} already exists. Target directory now '
                f'modified to {new_tr_dir}.'
            )
            tr_dir = new_tr_dir

    os.mkdir(tr_dir)
    return tr_dir


def save_log(
    log_dict: dict,
    dirpath: str,
    distributed: bool,
    device: torch.device
) -> None:
    '''
    Saves the given training stats log as a csv inside the specified directory.
    The specific way in which the saved file is named is determined from the
    distributed variable.
    '''
    
    assert os.path.isdir(dirpath)

    if distributed:
        destination = os.path.join(
            dirpath,
            f'training_log_{device}.csv'#.replace(':','_') # colons are illegal in windows filenames so 'cuda:0' is not allowed
        )
    else:
        destination = os.path.join(dirpath, f'training_log.csv')
    
    print_with_deviceid(f'Saving training log to {destination}...', device)
    log_df = pd.DataFrame.from_dict(data=log_dict, orient='index')
    log_df.to_csv(destination, index_label='epoch')


def save_model(
    model: torch.nn.Module,
    name: str,
    dirpath: str,
    distributed: bool,
    device: torch.device,
    test_mode: bool
) -> None:
    '''
    If appropriate, save the given model inside the given directory. Whether it
    is appropriate for the current process to save a copy of the model or not
    is determined based on the distributed, device and test_mode variables.
    '''

    assert os.path.isdir(dirpath)
    
    # Option 1: The script is in distributed mode and test mode, so each
    # process must save its own copy of the model with the filename needing to
    # distinguish copies of the model from different processes
    if distributed and test_mode:
        destination = os.path.join(
            dirpath,
            f'{name}_{device}.ptnn'#.replace(':', '_')
        )

        print_with_deviceid(
            f'Saving pretrained model to {destination}...',
            device
        )

        torch.save(model.module.cpu(), destination)
        return

    # Option 2: Save the model in the usual way (either the program is not
    # running in distributed mode, or if it is, it is the process with rank 0)
    if not distributed or (distributed and device.index == 0):
        destination = os.path.join(dirpath, f'{name}.ptnn')

        if distributed:
            model = model.module
        
        print_with_deviceid(
            f'Saving pretrained model to {destination}...',
            device
        )

        torch.save(model.cpu(), destination)
        return


def write_hyperparameters(hyperparameters: dict, dirpath: str) -> None:
    '''
    Write hyperparameters to a text file named 'hyperparams.txt' in the
    specified directory.
    '''

    destination = os.path.join(dirpath, "hyperparams.txt")

    print(f'Writing hyperparameters to {destination}...')

    with open(destination, 'w') as f:
        f.writelines([f'{k}: {hyperparameters[k]}\n' for k in hyperparameters])


# Hyperparameter parsing/loading
def bool_convert(x: str):
    '''
    Converts string to bool values, where 'True' is mapped to True, and 'False'
    is mapped to False. Helper function to hyperparams.
    '''

    if x == 'True':
        return True

    if x == 'False':
        return False
    
    raise RuntimeError(f'Expected "True" or "False", got {x}')


type_dict = {
    'bool': bool_convert,
    'float': float,
    'int': int,
    'str': str
}


def parse_hyperparams(csv_path: str) -> dict:
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

    if not (csv_path.endswith('.csv') and os.path.isfile(csv_path)):
        raise RuntimeError(f'Bad path to csv: {csv_path}.')

    df = pd.read_csv(csv_path)
    check_dataframe_format(df, ['param_name', 'type', 'value'])

    hyperparams = dict()
    for _, row in df.iterrows():
        if not row['type'] in type_dict:
            raise RuntimeError(f'Unrecognised data type: {row["type"]}')
        
        param_value = type_dict[row['type']](row['value'])
        hyperparams[row['param_name']] = param_value

    return hyperparams