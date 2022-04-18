from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase as Tokenizer, AutoTokenizer

from data.emqa_dataset import EmqaDataset


# noinspection PyAbstractClass
class EmqaDataModule(LightningDataModule):
    tokenizer: Tokenizer

    def __init__(self, config, drop_last=True):
        """

        :param config: Needs {tokenizer_name: str,
                              separate_question_tok_name: Optional[str],
                              drop_val_last: Optional[bool] = False
                              workers: int,
                              use_final_test: bool,
                              train_bsz: int,
                              test_bsz: int
                              } + requirements from EmqaDataset.create_from_cfg
        """
        super().__init__()
        self.config = config
        self.drop_last = drop_last
        self.drop_val_last = getattr(config, 'drop_val_last', False)
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set this per convenience for GPT-2
        if getattr(config, 'separate_question_tok_name', None):
            self.question_tok = AutoTokenizer.from_pretrained(config.separate_question_tok_name)
        else:
            self.question_tok = None

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self.train_dataset = self._create_data('train')
        self.val_dataset = self._create_data('val')
        if self.config.use_final_test:
            self.test_dataset = self._create_data('test')
        else:
            self.test_dataset = self.val_dataset or self._create_data('val')

    def _create_data(self, split):
        return EmqaDataset.create_from_cfg(self.config, split, self.tokenizer, self.question_tok)

    def common_loader_args(self, dataset):
        return dict(num_workers=self.config.workers,
                    collate_fn=dataset.collate_emv_samples,
                    pin_memory=True)

    def eval_loader_args(self, dataset):
        return dict(**self.common_loader_args(dataset),
                    shuffle=False,
                    batch_size=self.config.test_bsz)

    def train_dataloader(self):
        assert self.train_dataset
        return DataLoader(self.train_dataset,
                          batch_size=self.config.train_bsz,
                          shuffle=True,
                          drop_last=self.drop_last,
                          **self.common_loader_args(self.train_dataset))

    def val_dataloader(self):
        assert self.val_dataset
        return DataLoader(self.val_dataset, drop_last=self.drop_val_last,
                          **self.eval_loader_args(self.val_dataset))

    def test_dataloader(self):
        assert self.test_dataset
        return DataLoader(self.test_dataset, **self.eval_loader_args(self.test_dataset))
