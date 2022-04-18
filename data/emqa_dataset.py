import json
import math
from abc import ABC
from pathlib import Path
from typing import Dict, Union, List
from typing import Literal

import h5py
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase as Tokenizer

EmqaAnnotationFileDictKey = Literal[
    'sample_id',
    'video_id',
    'question',
    'answer',
    'moment_start_frame',  # optional
    'moment_end_frame',  # optional
]
EmqaSampleDictKey = Literal[
    'video_id',
    'sample_id',
    'question_text',
    'answer_text',
    'video_features',
    'moment_label'
]
EmqaBatchDictKey = Literal[
    'batch_video_ids',  # List[int]
    'batch_sample_ids',  # List[int]
    'batch_question_tokens',  # int, B x L
    'batch_question_mask',  # bool, B x L
    'batch_video_features',  # float, B x N x H
    'batch_video_mask',  # bool, B x N
    'batch_answer_texts',  # List[str]
    'batch_answer_tokens',  # int, B x T
    'batch_answer_mask',  # bool, B x T
    'batch_moment_localization_labels'  # float, B x N, optional
]
EmqaBatch = Dict[EmqaBatchDictKey, Union[torch.Tensor, List[int], List[str]]]
EmqaSample = Dict[EmqaSampleDictKey, Union[torch.Tensor, int, str]]
DatasetSplitName = Literal["train", "val", "test"]


class EmqaDataset(Dataset, ABC):
    tokenizer: Tokenizer
    split: DatasetSplitName

    def __init__(self, video_features_file: Path,
                 tokenizer: Tokenizer,
                 split: DatasetSplitName,
                 annotations: List[Dict[EmqaAnnotationFileDictKey, Union[int, str]]],
                 normalize_video_features=False,
                 frames_per_feature=16,
                 separate_question_tokenizer: Tokenizer = None):
        super().__init__()
        self.video_features_file = video_features_file
        self.tokenizer = tokenizer
        self.separate_question_tokenizer = separate_question_tokenizer or tokenizer
        self.split = split
        self.annotations = annotations
        self.normalize_video_features = normalize_video_features
        self.frames_per_feature = frames_per_feature

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index) -> EmqaSample:
        video_id = self.annotations[index]['video_id']
        sample_id = self.annotations[index]['sample_id']
        question = self.annotations[index]['question']
        answer = self.annotations[index]['answer']
        gt_start_frame = self.annotations[index].get('moment_start_frame')
        gt_end_frame = self.annotations[index].get('moment_end_frame')
        video_features = self._get_video_features(video_id)

        sample: EmqaSample = {
            'video_id': video_id,
            'sample_id': sample_id,
            'video_features': video_features,
            'question_text': question,
            'answer_text': answer
        }
        if gt_start_frame is not None and gt_end_frame is not None:
            start = gt_start_frame // self.frames_per_feature
            # ensure at least one target frame even if gt_start == gt_end
            end = math.ceil(gt_end_frame / self.frames_per_feature)
            if start == end:
                end += 1
            sample['moment_label'] = torch.tensor([start, end], dtype=torch.int)
        return sample

    def _get_video_features(self, video_id):
        with h5py.File(self.video_features_file, 'r') as hdf5_file:
            features = torch.from_numpy(hdf5_file[video_id][:]).float()
        if self.normalize_video_features:
            features = F.normalize(features, dim=1)
        return features

    def collate_emv_samples(self, batch: List[EmqaSample]) -> EmqaBatch:
        video_ids = [b['video_id'] for b in batch]
        sample_ids = [b['sample_id'] for b in batch]
        video_features = [b['video_features'] for b in batch]
        questions = [b['question_text'] for b in batch]
        answers = [b['answer_text'] for b in batch]

        answers_with_eos = [a + self.tokenizer.eos_token for a in answers]
        tok_args = dict(padding=True, return_tensors='pt', add_special_tokens=False)
        question_tok = self.separate_question_tokenizer(questions, **tok_args)
        answers_tok = self.tokenizer(answers_with_eos, **tok_args)

        video_features_padded = pad_sequence(video_features, batch_first=True)
        video_mask = pad_sequence([torch.ones(len(v)) for v in video_features], batch_first=True).bool()

        result: EmqaBatch = {
            'batch_video_ids': video_ids,
            'batch_sample_ids': sample_ids,
            'batch_question_tokens': question_tok['input_ids'],
            'batch_question_mask': question_tok['attention_mask'],
            'batch_answer_texts': answers,
            'batch_answer_tokens': answers_tok['input_ids'],
            'batch_answer_mask': answers_tok['attention_mask'],
            'batch_video_features': video_features_padded,
            'batch_video_mask': video_mask
        }
        if 'moment_label' in batch[0]:
            moment_labels = torch.zeros(len(batch), video_features_padded.shape[1])
            for i, b in enumerate(batch):
                gt_start, gt_end = b['moment_label']
                moment_labels[i, gt_start:gt_end] = 1
                # add smoothing before/after gt_start & end ?
            result['batch_moment_localization_labels'] = moment_labels

        return result

    @classmethod
    def create_from_cfg(cls, cfg, split: DatasetSplitName, tokenizer: Tokenizer,
                        separate_question_tok: Tokenizer = None):
        """
        Create EmqaDataset from cfg.

        :param cfg: Needs data_dir, feature_type
        :param split: train / val / test
        :param tokenizer: hugginface tokenizer
        :param separate_question_tok: separate tok for questions
        :return: EmqaDataset
        """
        ds_dir = Path(cfg.data_dir)
        video_features_file = ds_dir / f'{split}-{cfg.feature_type}.hdf5'
        if not video_features_file.is_file():
            video_features_file = ds_dir / f'{cfg.feature_type}.hdf5'
            if not video_features_file.is_file():
                raise ValueError(str(video_features_file))
        annotation_file = ds_dir / f'annotations.{split}.json'
        annotations = json.loads(annotation_file.read_text())
        return cls(video_features_file, tokenizer, split, annotations,
                   separate_question_tokenizer=separate_question_tok)
