import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any

from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenize import tokenize
from sacrebleu.metrics import BLEU, BLEUScore

from .util import AverageMeter


# Check whether to use
# - https://github.com/Maluuba/nlg-eval
# - https://github.com/hwanheelee1993/KPQA
def calc_metrics(predictions: List[str], gold_annotations: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate metrics.

    Parameters
    ----------
    predictions : list[str]
        The list of predictions
    gold_annotations : list[list[str]]
        A list with the same length as predictions.
        Each element is a list of possible target candidates for the corresponding prediction.
        All elements should have the same length.
    """
    if len(predictions) != len(gold_annotations):
        raise ValueError(f'{len(predictions)} != {len(gold_annotations)}')
    ref_count = len(gold_annotations[0])
    if any(len(refs) != ref_count for refs in gold_annotations):
        raise ValueError(f'All refs should have the same length {ref_count}!')

    acc = _calc_accuracy(predictions, gold_annotations)
    bleu = _calc_bleu(predictions, gold_annotations)
    rouge = _calc_rouge(predictions, gold_annotations)
    meteor = _calc_meteor(predictions, gold_annotations)

    return {
        'plain_acc': acc,
        **bleu,
        'ROUGE': rouge['rougeL']['f'],
        **_flatten_dict(rouge, prefix='ROUGE.'),
        'METEOR': meteor
    }


def _calc_accuracy(predictions, gold_annotations):
    correct = 0
    for pred, possible_refs in zip(predictions, gold_annotations):
        if any(ref == pred for ref in possible_refs):
            correct += 1
    total = len(predictions)
    return correct / total


def _calc_meteor(predictions, gold_annotations):
    score = AverageMeter()
    for pred, possible_refs in zip(predictions, gold_annotations):
        pred = tokenize(pred, None)
        # https://github.com/cmu-mtlab/meteor/blob/master/src/edu/cmu/meteor/util/Normalizer.java
        possible_refs = [tokenize(x, None) for x in possible_refs]
        score.update(meteor_score(possible_refs, pred))
    return score.avg


def _calc_rouge(predictions, gold_annotations) -> Dict[str, Dict[str, float]]:
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge = defaultdict(lambda: defaultdict(AverageMeter))
    for pred, possible_refs in zip(predictions, gold_annotations):
        sample_result = {}
        for ref in possible_refs:
            single_ref_result = rouge_scorer.score(ref, pred)
            for k, scores in single_ref_result.items():
                existing_result_dict = sample_result.setdefault(k, {})
                if existing_result_dict.get('f', -1) < scores.fmeasure:
                    existing_result_dict.update(f=scores.fmeasure, p=scores.precision, r=scores.recall)
        for k, best_scores in sample_result.items():
            rouge[k]['p'].update(best_scores['p'])
            rouge[k]['r'].update(best_scores['r'])
            rouge[k]['f'].update(best_scores['f'])
    return {
        rouge_type: {
            measure: score.avg
            for measure, score in results.items()
        } for rouge_type, results in rouge.items()
    }


def _calc_bleu(predictions, gold_annotations) -> Dict[str, float]:
    refs_transposed = [
        [refs[i] for refs in gold_annotations]
        for i in range(len(gold_annotations[0]))
    ]
    bleu: BLEUScore = BLEU().corpus_score(predictions, refs_transposed)
    return {
        'BLEU': bleu.score,
        'BLEU.bp': bleu.bp,
        'BLEU.ratio': bleu.ratio,
        'BLEU.hyp_len': float(bleu.sys_len),
        'BLEU.ref_len': float(bleu.ref_len),
    }


def _flatten_dict(d, prefix=''):
    result = {}
    for k, v in d.items():
        my_key = prefix + k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=my_key + '.'))
        else:
            result[my_key] = v
    return result


def main():
    parser = ArgumentParser('Eval output file')
    parser.add_argument('--gold_answers', type=str, required=True,
                        help='Path to answers.json, containing mapping from sample_id to answer')
    parser.add_argument('eval_file', type=str,
                        help='JSON File to evaluate. Should contain mapping from sample_id '
                             'to hypothesis or array of hypotheses')
    args = parser.parse_args()

    gold_answers = json.loads(Path(args.gold_answers).read_text())
    hypotheses = json.loads(Path(args.eval_file).read_text())
    if isinstance(next(iter(hypotheses.values())), list):
        hypotheses = {k: v[0] for k, v in hypotheses.items()}
    assert len(hypotheses.keys() - gold_answers.keys()) == 0, 'No gold answer for some hypotheses'

    gold_and_hypo = [(gold_answers[k], hypotheses[k]) for k in hypotheses.keys()]
    hypo_list = [h for g, h in gold_and_hypo]
    gold_list = [[g] for g, h in gold_and_hypo]
    metrics = calc_metrics(hypo_list, gold_list)

    pprint(metrics)


if __name__ == '__main__':
    main()
