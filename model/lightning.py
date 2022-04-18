import random
from itertools import chain
from typing import List, Dict, Union

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizerBase as Tokenizer

from data.emqa_dataset import EmqaBatch
from eval.eval import calc_metrics
from lightning_util import freeze_params
from model.base import EmqaBaseModel


# noinspection PyAbstractClass
class EmqaLightningModule(LightningModule):

    def __init__(self, model_config, optim_config, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.save_hyperparameters(dict(model=model_config, optim=optim_config))
        self.model: EmqaBaseModel = instantiate(model_config)
        self.optimizer_config = optim_config.optimizer
        if 'loss_weights' in optim_config:
            self.loss_weights: Dict[str, float] = optim_config.loss_weights
            self._loss_calc_cache = None
        else:
            self.loss_weights = None
        self.tokenizer = tokenizer
        self.lr = self.optimizer_config.lr
        freeze_params(self, optim_config.freeze)
        self._log_indices = {}

    def training_step(self, batch: EmqaBatch, batch_idx):
        loss_dict, logits = self.model(**self._get_model_inputs(batch))
        total_loss = self._modify_loss_dict(loss_dict)
        for k, v in loss_dict.items():
            self.log(k, v)
        return total_loss

    def _modify_loss_dict(self, loss_dict: Dict[str, torch.Tensor]):
        if 'total_loss' in loss_dict:
            return loss_dict['total_loss']
        if len(loss_dict) == 1:
            # No matter how it's called, the single value is the total loss
            #  However, no need to add it to loss_dict again (would only produce log twice)
            return next(iter(loss_dict.values()))
        assert self.loss_weights is not None
        if self._loss_calc_cache:
            master_key, remaining_keys = self._loss_calc_cache
        else:
            specified_keys = self.loss_weights.keys()
            all_keys = loss_dict.keys()
            master_keys = all_keys - specified_keys
            assert len(master_keys) == 1, f'There must be exactly one loss weight not specified (master weight), ' \
                                          f'got {all_keys} - {specified_keys} = {master_keys}'
            remaining_keys = all_keys - master_keys
            assert specified_keys == remaining_keys, f'{specified_keys} != {remaining_keys}'
            master_key = next(iter(master_keys))
            self._loss_calc_cache = (master_key, remaining_keys)
        master_loss = loss_dict[master_key]
        total_loss = master_loss + sum(self.loss_weights[k] * loss_dict[k] for k in remaining_keys)
        loss_dict['total_loss'] = total_loss  # Add it to dict for logging purposes
        return total_loss

    @staticmethod
    def _get_model_inputs(batch: EmqaBatch):
        return dict(
            question_tokens=batch['batch_question_tokens'],
            question_mask=batch['batch_question_mask'],
            video_features=batch['batch_video_features'],
            video_mask=batch['batch_video_mask'],
            answer_tokens=batch['batch_answer_tokens'],
            answer_mask=batch['batch_answer_mask'],
            batch_sample_ids=batch['batch_sample_ids'],
            moment_localization_labels=batch.get('batch_moment_localization_labels')
        )

    def validation_step(self, batch: EmqaBatch, batch_idx):
        loss_dict, lm_logits = self.model(**self._get_model_inputs(batch))
        self._modify_loss_dict(loss_dict)
        hypo_answers = self._extract_answers(lm_logits)
        return {'hypos': hypo_answers,
                'targets': batch['batch_answer_texts'],
                **loss_dict}

    def _extract_answers(self, lm_logits):
        sequences = lm_logits.argmax(dim=-1)
        hypo_answers = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return hypo_answers

    def validation_epoch_end(self, outputs: List[Dict[str, Union[torch.Tensor, List[str]]]]) -> None:
        def _mean(key):
            return torch.stack([data[key] for data in outputs]).mean()

        self._log_some_outputs(outputs, 'val')
        metrics = self.aggregate_metrics(outputs, prefix='val')
        metrics.update({
            f'val_{name}': _mean(name) for name in outputs[0].keys() if 'loss' in name
        })
        self.log_dict(metrics)

    def _log_some_outputs(self, outputs, name):
        num_val_steps_to_log, num_samples_per_batch_to_log = 5, 3  # Could be configurable via cfg
        if name in self._log_indices:
            steps_to_log_indices = self._log_indices[name]['steps']
        else:
            steps_to_log_indices = random.sample(range(len(outputs)), k=min(len(outputs), num_val_steps_to_log))
            self._log_indices[name] = {'steps': steps_to_log_indices, 'samples': [
                random.sample(range(len(outputs[step]['targets'])),
                              k=min(len(outputs[step]['targets']), num_samples_per_batch_to_log))
                for step in steps_to_log_indices
            ]}
        for i, step in enumerate(steps_to_log_indices):
            output, target = outputs[step]['hypos'], outputs[step]['targets']
            indices = self._log_indices[name]['samples'][i]
            for b in indices:
                sample = (
                    f'Target: "{target[b]}". \n'
                    f'Output: "{output[b]}"'
                )
                self.logger.experiment.add_text(f'{name} {str(i * len(indices) + b)}', sample,
                                                global_step=self.global_step)

    @staticmethod
    def aggregate_metrics(outputs, prefix):
        all_hypos = list(chain(*(data['hypos'] for data in outputs)))
        all_targets = list(chain(*(data['targets'] for data in outputs)))
        metrics = calc_metrics(all_hypos, [[x] for x in all_targets])
        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        return metrics

    def test_step(self, batch: EmqaBatch, batch_idx):
        sequences = self.model(**self._get_model_inputs(batch), teacher_forcing=False)
        hypo_answers = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return {'hypos': hypo_answers,
                'targets': batch['batch_answer_texts']}

    def test_epoch_end(self, outputs: List[Dict[str, Union[List[str], Dict]]]) -> None:
        self._log_some_outputs(outputs, 'test')
        metrics = self.aggregate_metrics(outputs, prefix='test')
        self.log_dict(metrics)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return instantiate(self.optimizer_config,
                           params,
                           # lr might be overridden by auto lr tuning
                           lr=self.lr)
