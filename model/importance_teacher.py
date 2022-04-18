from argparse import ArgumentParser

import torch
from hydra import initialize, compose
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from data.datamodule import EmqaDataModule
from data.emqa_dataset import EmqaBatch
from model.base import MemoryAugmentedTransformerEmqaModel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# noinspection PyAbstractClass
class ImportanceTeacherVqaModel(MemoryAugmentedTransformerEmqaModel):
    # Actually this is not really a MemoryAugmentedTransformerEmqaModel since it uses full attention over the input

    def __init__(self, fragment_length: int,
                 input_size: int,
                 pretrained_enc_dec: str) -> None:
        super().__init__(pretrained_enc_dec)
        self.clip_avg = nn.AvgPool1d(fragment_length, stride=1)
        self.question_embedding = self.transformer.get_encoder()

        hidden = self.transformer.get_input_embeddings().embedding_dim
        self.query_attn = nn.MultiheadAttention(embed_dim=hidden,
                                                num_heads=1, batch_first=True)

        sentence_emb = self.question_embedding.get_input_embeddings().embedding_dim
        if sentence_emb != hidden:
            self.transform_query = nn.Linear(sentence_emb, hidden, bias=False)
        else:
            self.transform_query = nn.Identity()
        if input_size != hidden:
            self.transform_visual = nn.Linear(input_size, hidden, bias=False)
        else:
            self.transform_visual = nn.Identity()

    def forward_encoders(self, question_tokens, question_mask, video_features, video_mask, moment_localization_labels):
        attn_context = self.attend_to_fragments(question_tokens, question_mask, video_features, video_mask)
        attn_mask = torch.ones(attn_context.shape[:-1], device=attn_context.device, dtype=torch.bool)

        context, context_mask = self._prepare_context(attn_context, attn_mask, question_tokens, question_mask)
        return context, context_mask, attn_context, attn_mask, {}

    def attend_to_fragments(self, question_tokens, question_mask,
                            visual_input, visual_mask,
                            return_attn_scores=False):
        # visual input: B x N x H. Want to avg pool over N => permute
        fragments = self.clip_avg(visual_input.permute(0, 2, 1))
        fragments = fragments.permute(0, 2, 1)  # permute back to B x N' x H
        # bsz x num_fragments : True <==> valid entry
        fragments_mask = (self.clip_avg(visual_mask.to(dtype=torch.float)) == 1).squeeze()

        # modify samples where fragment_length > input_length with correct averaging over non-padded values only
        # so that even short samples are represented by at least one fragment
        input_lengths = visual_mask.sum(dim=1)
        num_fragments = fragments_mask.sum(dim=1)
        too_short_videos = num_fragments == 0
        for b in too_short_videos.nonzero():
            b = b.item()
            fragments[b, 0] = visual_input[b, visual_mask[b]].sum(dim=0) / input_lengths[b]
            fragments_mask[b, 0] = 1
            num_fragments[b] = 1

        with torch.no_grad():
            model_output = self.question_embedding(question_tokens, question_mask)
            query = mean_pooling(model_output, question_mask)

        fragments = self.transform_visual(fragments)
        query = self.transform_query(query).unsqueeze(dim=1)  # add fake seq dimension

        # attn_mask: batch x querys x keys = bsz x 1 x num_fragments.
        #   True <==> corresponding position is _not_ allowed to attend
        attn_mask = ~fragments_mask[:, None, :]
        context_token, attn_weights = self.query_attn(query, fragments, fragments, attn_mask=attn_mask,
                                                      need_weights=return_attn_scores)  # B x 1 x H
        if return_attn_scores:
            # trim padding indices away
            attn_weights = attn_weights.squeeze()
            return context_token, [w[:length] for w, length in zip(attn_weights, num_fragments)]
        else:
            return context_token


def _extract_attn_scores(model: ImportanceTeacherVqaModel, train_iterator):
    all_scores = {}
    batch: EmqaBatch
    for batch in train_iterator:
        _, attn_scores = model.attend_to_fragments(
            batch['batch_question_tokens'].cuda(),
            batch['batch_question_mask'].cuda(),
            batch['batch_video_features'].cuda(),
            batch['batch_video_mask'].cuda(),
            return_attn_scores=True
        )
        for idx, scores in zip(batch['batch_sample_ids'], attn_scores):
            all_scores[idx] = scores.cpu()

    return all_scores


@torch.no_grad()
def main(output_file: str, checkpoint_path: str, config: DictConfig):
    checkpoint = torch.load(checkpoint_path)
    state_dict = {k[len('model.'):]: v for k, v in checkpoint['state_dict'].items()}
    model_cfg = dict(config.model)
    model_cfg.pop('_target_')
    model = ImportanceTeacherVqaModel(**model_cfg)
    # noinspection PyTypeChecker
    model.load_state_dict(state_dict)

    data = EmqaDataModule(config.dataset, drop_last=False)
    data.prepare_data()
    data.setup()

    model.cuda()
    result = {}
    with tqdm(data.train_dataloader(), 'Train') as train_iterator:
        result.update(_extract_attn_scores(model, train_iterator))
    with tqdm(data.val_dataloader(), 'Val') as val_iterator:
        result.update(_extract_attn_scores(model, val_iterator))

    torch.save(result, output_file)


def cli_main():
    parser = ArgumentParser(description='Save attention scores from ImportanceTeacherVqaModel')

    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--output_file', type=str, required=True, help='Where to save attention scores.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Where to load trained model from. PyTorch Lighting checkpoint from SimpleVqa training')
    args = parser.parse_args()

    initialize(config_path='../config', job_name="save_vqa_attn_scores")
    config = compose(config_name='base', overrides=[f'dataset.train_bsz={args.bsz}', f'dataset.test_bsz={args.bsz}'])
    main(args.output_file, args.checkpoint_path, config)


if __name__ == '__main__':
    cli_main()
