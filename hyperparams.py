hyperparams = {
    'path_train_data': 'data/train.csv',
    'path_valid_data': 'data/test.csv',
    'num_encoder_layers': 16,
    'd_model': 16,
    'nhead': 4,
    'dim_feedforward': 128,
    'batch_size': 256,
    'lr_scheduling': True,
    'lr': 0.001,
    'optim_warmup': 10_000,
    'num_epochs': 20,
}