from argparse import ArgumentParser

import hydra
from omegaconf import DictConfig

from data.datamodule import EmqaDataModule
from hydra_compat import apply_argparse_defaults_to_hydra_config
from lightning_util import tune_fit_test, add_common_trainer_util_args
from model.lightning import EmqaLightningModule


@hydra.main(config_path='config', config_name='base')
def main(config: DictConfig):
    fake_parser = ArgumentParser()
    add_common_trainer_util_args(fake_parser, default_monitor_variable='val_total_loss')
    apply_argparse_defaults_to_hydra_config(config.trainer, fake_parser)

    data = EmqaDataModule(config.dataset)
    model = EmqaLightningModule(config.model, config.optim, data.tokenizer)

    tune_fit_test(config.trainer, model, data)


if __name__ == '__main__':
    main()
