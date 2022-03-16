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
    'pretrain_id': '22-03-16-02:46',
    'path_train_data': 'data/labelled_train.csv',
    'path_valid_data': 'data/labelled_test.csv',
    'train_batch_size': 2048,
    'valid_batch_size': 512,
    'lr_scheduling': True,
    'lr': 0.001,
    'optim_warmup': 5,
    'num_epochs': 10,
}