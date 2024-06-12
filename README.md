# tcrlm

This repository contains code used to train the various SCEPTR variants published in this [preprint](https://arxiv.org/abs/2406.06397).
The entry point for starting a training run is by running the `train.py` script as an executable.

## What's a training run config json file?

> [!TIP]
> Much of the actual code relating to the architecture modules that the config references is in the [libtcrlm](https://github.com/yutanagano/libtcrlm) repo, where the actual model components are implemented.

When building the codebase for tcrlm/[libtcrlm](https://github.com/yutanagano/libtcrlm), I wanted to keep different components of the model very easy hot-swappable so that I could do fast trial-and-error for many different variants of the model.
So the general way the training pipeline works is you invoke `train.py` with a json file that is basically a list of configuration options that you want to set for your new training run, including things like:

- Model architecture settings (hyperparameters, which model components to use, which variant of the embedding system to use, etc)
- Location of your training/validation data files
- Data hyperparameters (batch size, number of epochs, what type of dataloader to use, etc.)
- Training hyperparameters, like learning rate, MLM/autocontrastive loss variants, hyperparams, etc.

Once you have a json file that correctly declares all of these settings you want, you just run

```bash
train.py <config_path>
```
  
where `<config_path>` is just the location of where the json file is.

An example of what a config should look like is given in `example_config.json`.
