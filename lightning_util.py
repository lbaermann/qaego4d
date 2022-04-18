import os
import re
import sys
from argparse import Namespace
from copy import deepcopy
from fnmatch import fnmatchcase
from pathlib import Path
from typing import List

import torch.nn
from pytorch_lightning import seed_everything, Trainer, Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin


def dict_parser(s: str):
    return eval('{' + re.sub(r'(\w+)=(["\']?\w+["\']?)', r'"\1":\2', s) + '}')


def add_common_trainer_util_args(parser, default_monitor_variable='val_loss', default_monitor_mode='min'):
    if default_monitor_mode not in ['min', 'max']:
        raise ValueError(default_monitor_mode)
    parser.add_argument('--lr_find_kwargs', default=dict(min_lr=5e-6, max_lr=1e-2), type=dict_parser,
                        help='Arguments for LR find (--auto_lr_find). Default "min_lr=5e-6,max_lr=1e-2"')
    parser.add_argument('--random_seed', default=42, type=lambda s: None if s == 'None' else int(s),
                        help='Seed everything. Set to "None" to disable global seeding')
    parser.add_argument('--auto_resume', default=False, action='store_true',
                        help='Automatically resume last saved checkpoint, if available.')
    parser.add_argument('--test_only', default=False, action='store_true',
                        help='Skip fit and call only test. This implies automatically detecting newest checkpoint, '
                             'if --checkpoint_path is not given.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Load this checkpoint to resume training or run testing. '
                             'Pass in the special value "best" to use the best checkpoint according to '
                             'args.monitor_variable and args.monitor_mode. '
                             'Using "best" only works with test_only mode.')
    parser.add_argument('--ignore_existing_checkpoints', default=False, action='store_true',
                        help='Proceed even with training a new model, even if previous checkpoints exists.')
    parser.add_argument('--monitor_variable', default=default_monitor_variable, type=str,
                        help='Variable to monitor for early stopping and for checkpoint selection. '
                             f'Default: {default_monitor_variable}')
    parser.add_argument('--monitor_mode', default=default_monitor_mode, type=str, choices=['min', 'max'],
                        help='Mode for monitoring the monitor_variable (for early stopping and checkpoint selection). '
                             f'Default: {default_monitor_mode}')
    parser.add_argument('--reset_early_stopping_criterion', default=False, action='store_true',
                        help='Reset the early stopping criterion when loading from checkpoint. '
                             'Prevents immediate exit after switching to more complex dataset in curriculum strategy')


def _auto_resume_from_checkpoint(args):
    if getattr(args, 'resume_from_checkpoint', None) is not None:
        raise DeprecationWarning('Trainer.resume_from_checkpoint is deprecated. Switch to checkpoint_path argument.')
    best_mode = args.checkpoint_path == 'best'
    if best_mode:
        if not args.test_only:
            raise RuntimeError('checkpoint_path="best" only works in test_only mode!')
        # More "best" logic is handled below
    elif args.checkpoint_path is not None:
        return

    log_dir = Path(getattr(args, 'default_root_dir', None) or 'lightning_logs')
    existing_checkpoints = list(log_dir.glob('version_*/checkpoints/*.ckpt'))
    if len(existing_checkpoints) == 0:
        return  # This is the first run
    if not args.test_only and not args.auto_resume:
        if args.ignore_existing_checkpoints:
            return  # Explicitly requested
        raise RuntimeWarning(f"There already exist checkpoints, but checkpoint_path/auto_resume not set! "
                             f"{existing_checkpoints}")
    if best_mode:
        chosen = _auto_choose_best_checkpoint(args, existing_checkpoints)
    else:
        chosen = _auto_choose_newest_checkpoint(existing_checkpoints)
    args.checkpoint_path = str(chosen)
    print(f'Auto-detected {"best" if best_mode else "newest"} checkpoint {chosen}, resuming it... '
          f'If this is not intended, use --checkpoint_path !')


def _auto_choose_newest_checkpoint(existing_checkpoints):
    chosen = None
    for c in existing_checkpoints:
        if chosen is None or c.stat().st_mtime > chosen.stat().st_mtime:
            chosen = c
    return chosen


def _auto_choose_best_checkpoint(args, existing_checkpoints):
    chosen = None
    for c in existing_checkpoints:
        if chosen is None or 'last.ckpt' == chosen.name:
            chosen = c
            continue
        if 'last.ckpt' == c.name:
            continue
        chosen_match = re.search(fr'{re.escape(args.monitor_variable)}=(\d+(?:\.\d+)?)', chosen.name)
        current_match = re.search(fr'{re.escape(args.monitor_variable)}=(\d+(?:\.\d+)?)', c.name)
        if chosen_match is None:
            raise ValueError(chosen)
        if current_match is None:
            raise ValueError(c)
        op = {'min': lambda old, new: new < old, 'max': lambda old, new: new > old}[args.monitor_mode]
        if op(float(chosen_match.group(1)), float(current_match.group(1))):
            chosen = c
    return chosen


def apply_common_train_util_args(args) -> List[Callback]:
    _auto_resume_from_checkpoint(args)
    if args.random_seed is not None:
        seed_everything(args.random_seed, workers=True)

    early_stopping = EarlyStopping(monitor=args.monitor_variable, mode=args.monitor_mode,
                                   min_delta=0.001, patience=10,
                                   check_on_train_epoch_end=False)
    if args.reset_early_stopping_criterion:
        # Prevent loading the early stopping criterion when restoring from checkpoint
        early_stopping.on_load_checkpoint = lambda *args, **kwargs: None
    return [
        early_stopping,
        ModelCheckpoint(save_last=True, monitor=args.monitor_variable, mode=args.monitor_mode,
                        save_top_k=1, filename='{step}-{' + args.monitor_variable + ':.3f}')
    ]


def _ddp_save_tune(args, model, datamodule):
    """
    Runs LR tuning on main process only, _before_ DDP is initialized. Sets env var for communication to child processes.
    """
    lr_env_var = os.getenv('_pl_auto_lr_find')
    if lr_env_var is None:
        # Main process running this code will not have a value for the env var, and thus perform single-process LR tune
        args_copy = deepcopy(args)
        args_copy.strategy = None
        args_copy.gpus = 1
        print('Running single GPU tune...')
        single_process_trainer = Trainer.from_argparse_args(args_copy)
        single_process_trainer.tune(model, datamodule, lr_find_kwargs=dict(args.lr_find_kwargs))
        # "Broadcast" result to other ranks. Can not use actual broadcast mechanism,
        # since DDP env is not yet running.
        os.environ['_pl_auto_lr_find'] = str(model.lr)
    else:
        # Later workers executing this code will just load the best LR from the env var
        model.lr = float(os.environ['_pl_auto_lr_find'])


def _adjust_ddp_config(trainer_cfg):
    trainer_cfg = dict(trainer_cfg)
    strategy = trainer_cfg.get('strategy', None)
    if trainer_cfg['gpus'] > 1 and strategy is None:
        strategy = 'ddp'  # Select ddp by default
    if strategy == 'ddp':
        trainer_cfg['strategy'] = DDPPlugin(find_unused_parameters=False, gradient_as_bucket_view=True)
    return trainer_cfg


def tune_fit_test(trainer_cfg, model, datamodule):
    callbacks = apply_common_train_util_args(trainer_cfg)

    trainer_cfg = Namespace(**_adjust_ddp_config(trainer_cfg))
    ddp_mode = isinstance(getattr(trainer_cfg, 'strategy', None), DDPPlugin) and trainer_cfg.gpus > 1
    if (ddp_mode
            and not trainer_cfg.test_only
            and trainer_cfg.checkpoint_path is None
            and trainer_cfg.auto_lr_find):
        # Do tuning with a "fake" trainer on single GPU
        _ddp_save_tune(trainer_cfg, model, datamodule)

    trainer = Trainer.from_argparse_args(trainer_cfg, callbacks=[
        LearningRateMonitor(logging_interval='step'),
        *callbacks
    ])

    if not trainer_cfg.test_only:
        if trainer_cfg.checkpoint_path is None and not ddp_mode:
            # Do tune with the trainer directly
            trainer.tune(model, datamodule, lr_find_kwargs=dict(trainer_cfg.lr_find_kwargs))

        trainer.fit(model, datamodule, ckpt_path=trainer_cfg.checkpoint_path)

    trainer.test(
        model, datamodule,
        ckpt_path=
        trainer_cfg.checkpoint_path if trainer_cfg.test_only
        else None  # If fit was called before, we should _not_ restore initial train checkpoint
    )

    if not trainer_cfg.test_only and not trainer.fast_dev_run:
        # test call above runs test on last after training,
        # also want it on best checkpoint by default
        # before, flush stdout and err to avoid strange order of outputs
        print('', file=sys.stdout, flush=True)
        print('', file=sys.stderr, flush=True)
        trainer.test(model, datamodule, ckpt_path='best')


def freeze_params(model: torch.nn.Module, freeze_spec: List[str]):
    """
    Freeze parameters that begin with any of the freeze_spec items or match any of them according to fnmatchcase.

    Parameters
    ----------
    model       the model
    freeze_spec specifies which parameters to freeze (e.g. 'generator' or 'generator.h.*.weight')
    """
    for name, p in model.named_parameters():
        freeze = any(
            name.startswith(pattern) or fnmatchcase(name, pattern)
            for pattern in freeze_spec
        )
        if freeze:
            p.requires_grad = False
