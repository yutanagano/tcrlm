'''
training.py
purpose: Python module with helper classes for training CDR3Bert.
author: Yuta Nagano
ver: 3.1.0
'''


import os
import pandas as pd
import shutil
import torch


# Helper dictionary for parse_hyperparams
def bool_convert(x: str):
    '''
    Converts string to bool values, where 'True' is mapped to True, and 'False'
    is mapped to False. Helper function to hyperparams.
    '''
    if x == 'True':
        return True
    elif x == 'False':
        return False
    else:
        raise RuntimeError(f'Unexpected value: {x} (must be "True" or "False")')


type_dict = {
    'bool': bool_convert,
    'float': float,
    'int': int,
    'str': str
}


# Functions
def compare_models(dirpath: str, n_gpus: int) -> None:
    '''
    Compare all models saved from distributed processes and ensure that they
    are all equivalent i.e. that they all have the same weights.
    '''
    print('Comparing model enpoints from distributed training...')

    # Load the models
    paths_to_models = [
        os.path.join(dirpath, file) for file in os.listdir(dirpath) \
        if file.endswith('.ptnn')
    ]

    assert(len(paths_to_models) == n_gpus)

    # User feedback
    print('Model endpoints detected:')
    for path in paths_to_models: print(path)

    models = [torch.load(path) for path in paths_to_models]

    # Compare the models
    def compare_two(model1: torch.nn.Module, model2: torch.nn.Module):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.equal(p1, p2): return False
        return True
    
    for i, model2 in enumerate(models[1:], start=1):
        if not compare_two(models[0], model2):
            raise RuntimeError(
                f'compare_models(): models {paths_to_models[0]} and '\
                f'{paths_to_models[i]} are not equivalent.'
            )
    
    # Comparisons finished, no errors found!
    print('All model endpoints compared, no discrepancies found!')


def create_training_run_directory(
    run_id: str,
    mode: str,
    overwrite: bool = False
) -> str:
    '''
    Creates a directory for a training run with the specified run ID. The
    directory will be created in the 'pretrain_runs' parent directory if mode is
    set to 'pretrain', and in the 'finetune_runs' parent directory if mode is
    set to 'finetune'. If the relevant parent directory does not yet exist in
    the working directory, create it first, and then create a training run
    directory inside it. Return a string path to the newly created training run
    directory. If 'overwrite' is specified and a training run directory of the
    same name/run ID already exists, it and its contents will be deleted/
    overwritten. Otherwise, the existence of a similarly named directory will
    cause the program to systematically search for an alternative and available
    run ID.
    '''
    # Establish the name of the relevant parent directory
    if mode == 'pretrain': parent_dir = 'pretrain_runs'
    elif mode == 'finetune': parent_dir = 'finetune_runs'
    else:
        raise RuntimeError(
            f'mode value of {mode} not recognised- valid values are either '
            '"pretrain" or "finetune".'
        )

    # Create parent directory if not already existent
    if not os.path.isdir(parent_dir):
        print(f'Creating new parent directory {parent_dir}...')
        os.mkdir(parent_dir)

    # Create a path to a target directory corresponding to the specified run_id
    tr_dir = os.path.join(parent_dir,run_id)

    # If there already exists a directory at the specified path/name
    if os.path.isdir(tr_dir):
        # If overwrite=True, delete the preexisting directory along with all
        # contents, to free up that path address.
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
                f'A directory {tr_dir} already exists. Target directory now '\
                f'modified to {new_tr_dir}.'
            )
            tr_dir = new_tr_dir

    # Create the directory
    os.mkdir(tr_dir)

    # Return the path to that directory
    return tr_dir


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
    # Check that the provided csv path is legitimate
    if not csv_path.endswith('.csv') or not os.path.isfile(csv_path):
        raise RuntimeError('Please provide a valid path to a csv file.')

    # Read the csv file
    df = pd.read_csv(csv_path)

    # Check the header to ensure that it respects the format
    if df.columns.tolist() != ['param_name', 'type', 'value']:
        raise RuntimeError(
            f'The csv file has an incorrect format ({df.columns.tolist()}). '
            'The column names must be: ["param_name", "type", "value"].'
        )

    # Iterate over the csv and process each hyperparameter, save into dict
    hyperparams = dict()

    for idx, row in df.iterrows():
        if not row['type'] in type_dict:
            raise RuntimeError(f'Unrecognised data type: {row["type"]}')
        
        param_value = type_dict[row['type']](row['value'])
        hyperparams[row['param_name']] = param_value

    return hyperparams


def print_with_deviceid(msg: str, device: torch.device) -> None:
    print(f'[{device}]: {msg}')


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
    # First, ensure that the specified directory exists
    assert(os.path.isdir(dirpath))

    # Establish what the destination path for the saved file is
    if distributed:
        destination = os.path.join(dirpath, f'training_log_{device}.csv')
    else:
        destination = os.path.join(dirpath, f'training_log.csv')
    print_with_deviceid(f'Saving training log to {destination}...', device)

    # Save the log data as a csv at the destination
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
    is appropriate for the current process to save a copy of the model or not is
    determined based on the distributed, device and test_mode variables.
    '''
    # First, ensure that the specified directory exists
    assert(os.path.isdir(dirpath))
    
    # Option 1: The script is in distributed mode and test mode, so each
    # process must save its own copy of the model with the filename needing to
    # distinguish copies of the model from different processes
    if distributed and test_mode:
        # Establish the destination path
        destination = os.path.join(dirpath, f'{name}_{device}.ptnn')

        # As we are running in distributed mode, the DDP object must be
        # unwrapped, before saving.
        print_with_deviceid(
            f'Saving pretrained model to {destination}...',
            device
        )
        torch.save(model.module.cpu(), destination)

    # Option 2: Save the model in the usual way (either the program is not
    # running in distributed mode, or if it is, it is the process with rank 0)
    elif not distributed or (distributed and device.index == 0):
        # Establish the destination path
        destination = os.path.join(dirpath, f'{name}.ptnn')

        # Unwrap DistributedDataParallel if necessary
        if distributed: model = model.module

        # Save wrapped (with Cdr3PretrainWrapper) model
        print_with_deviceid(
            f'Saving pretrained model to {destination}...',
            device
        )
        torch.save(model.cpu(), destination)


def set_env_vars(master_addr: str, master_port: str) -> None:
    '''
    Set some environment variables that are required when spawning parallel
    processes using torch.nn.parallel.DistributedDataParallel.
    '''
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port


def write_hyperparameters(hyperparameters: dict, dirpath: str) -> None:
    '''
    Write hyperparameters to a text file named 'hyperparams.txt' in the
    specified directory.
    '''
    print(
        f'Writing hyperparameters to '\
        f'{os.path.join(dirpath, "hyperparams.txt")}...'
    )
    with open(os.path.join(dirpath, 'hyperparams.txt'), 'w') as f:
        f.writelines([f'{k}: {hyperparameters[k]}\n' for k in hyperparameters])


# Classes
class AdamWithScheduling:
    '''
    Wrapper around optimiser to implement custom learning rate scheduling.
    '''
    def __init__(
        self,
        params,
        d_model: int,
        n_warmup_steps: int,
        lr: float = 0.001,
        betas: (float, float) = (0.9, 0.999),
        eps: float = 1e-08,
        lr_multiplier: float = 1,
        scheduling: bool = True,
        decay: bool = True
    ):
        self.optimiser = torch.optim.Adam(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps
        )
        self._lr_multiplier = lr_multiplier
        self._d_model = d_model
        self._n_warmup_steps = n_warmup_steps
        self._scheduling = scheduling
        self._decay = decay

        self._step_num = 1
        self._lr_explicit = lr


    @property
    def step_num(self) -> int:
        return self._step_num


    @property
    def lr(self) -> float:
        if self._scheduling:
            return self.calculate_lr(self._step_num)
        else:
            return self._lr_explicit


    def step(self) -> None:
        # Update learning rate and step with the inner optimiser
        if self._scheduling: self._update_lr()
        self.optimiser.step()
        self._step_num += 1


    def zero_grad(self) -> None:
        # Zero out gradients with inner optimiser
        self.optimiser.zero_grad()
    

    def calculate_lr(self, step_num: int) -> float:
        # Learning rate decays inversely with the square root of step number
        if self._decay:
            return self._lr_multiplier * self._d_model ** -0.5 * \
                min(
                    step_num ** (-0.5),
                    step_num * self._n_warmup_steps ** (-1.5)
                )
        # Learning rate reaches target and stays there
        else:
            return min(
                self._lr_explicit,
                step_num / self._n_warmup_steps * self._lr_explicit
            )


    def _update_lr(self) -> None:
        # Update the learning rate of the inner optimiser
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = self.lr