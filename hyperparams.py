pretrain_hyperparams = {
    'path_train_data': 'data/train.csv',
    'path_valid_data': 'data/test.csv',
    'num_encoder_layers': 16,
    'd_model': 128,
    'nhead': 8,
    'dim_feedforward': 1024,
    'activation': 'gelu',
    'train_batch_size': 2048,
    'valid_batch_size': 512,
    'batch_optimisation': False,
    'lr_scheduling': True,
    'lr': 0.001,
    'optim_warmup': 30_000,
    'num_epochs': 100,
}

finetune_hyperparams = {
    'pretrain_id': 'test',
    'path_train_data': os.path.join(
        'tests', 'data', 'mock_labelled_data.csv'),
    'path_valid_data': os.path.join(
        'tests', 'data', 'mock_labelled_data.csv'),
    'train_batch_size': 2048,
    'valid_batch_size': 512,
    'lr_scheduling': True,
    'lr': 0.001,
    'optim_warmup': 5,
    'num_epochs': 3,
}