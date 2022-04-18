import argparse
import json
from pathlib import Path
from typing import Dict


def convert(nlq_file: Path, answers: Dict[str, str]):
    """
    Reads NLQ JSON file and answer map, and creates a single dict with a list of
    {"video_id", "sample_id", "question", "answer", "moment_start_frame", "moment_end_frame"} objects.

    :param nlq_file: NLQ JSON file
    :param answers: Answer map
    """
    result = []

    annotations = json.loads(nlq_file.read_text())
    for video in annotations['videos']:
        for clip in video['clips']:
            for annotation in clip['annotations']:
                for i, query in enumerate(annotation['language_queries']):
                    if 'query' not in query or not query['query']:
                        continue
                    question = query['query'].replace('\n', '').replace(',', '').strip()
                    video_id = clip['clip_uid']
                    sample_id = f'{annotation["annotation_uid"]}_{i}'
                    if sample_id not in answers:
                        continue
                    answer = answers[sample_id].replace('\n', '').replace(',', '').strip()
                    fps = 30  # fps = 30 is known for canonical Ego4D clips
                    start_frame = query['clip_start_sec'] * fps
                    end_frame = query['clip_end_sec'] * fps

                    result.append({
                        'video_id': video_id,
                        'sample_id': sample_id,
                        'answer': answer,
                        'question': question,
                        'moment_start_frame': start_frame,
                        'moment_end_frame': end_frame
                    })

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ego4d', required=True, type=str,
                        help='Directory where you placed the Ego4D download.')
    parser.add_argument('--qaego4d', required=True, type=str,
                        help='Path to QaEgo4D answers.json file')
    args = parser.parse_args()

    ego4d_dir = Path(args.ego4d)
    ego4d_annotations_dir = ego4d_dir / 'v1' / 'annotations'
    qaego4d_file = Path(args.qaego4d)
    assert ego4d_dir.is_dir()
    assert ego4d_annotations_dir.is_dir()
    assert qaego4d_file.is_file()

    qaego4d_data = json.loads(qaego4d_file.read_text())
    nlq_train, nlq_val = [ego4d_annotations_dir / f'nlq_{split}.json' for split in ('train', 'val')]

    train = convert(nlq_train, qaego4d_data['train'])
    val = convert(nlq_val, qaego4d_data['val'])
    test = convert(nlq_val, qaego4d_data['test'])

    output_dir = qaego4d_file.parent
    (output_dir / 'annotations.train.json').write_text(json.dumps(train))
    (output_dir / 'annotations.val.json').write_text(json.dumps(val))
    (output_dir / 'annotations.test.json').write_text(json.dumps(test))


if __name__ == '__main__':
    main()
