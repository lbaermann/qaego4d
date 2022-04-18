import json
import math
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, Manager, JoinableQueue
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm


def existing_path(p: str):
    p = Path(p)
    assert p.exists(), p
    return p


@dataclass
class WorkItem:
    video_uid: str
    clip_uid: str
    clip_start_frame: int
    clip_end_frame: int

    def output_file(self, output_dir: Path):
        return output_dir / f'{self.clip_uid}.pt'

    def do_work(self, video_features_dir: Path, output_dir: Path, feature_window_stride: int,
                progress_queue: JoinableQueue):
        video_features_file = video_features_dir / f'{self.video_uid}.pt'
        output_file = self.output_file(output_dir)
        start_feature = self.clip_start_frame // feature_window_stride
        end_feature = math.ceil(self.clip_end_frame / feature_window_stride)
        video_features = torch.load(str(video_features_file))
        clip_features = video_features[start_feature:end_feature]
        torch.save(clip_features, str(output_file))
        progress_queue.put_nowait(1)


def _extract_work_items(annotations) -> List[WorkItem]:
    work_items = []
    for video in annotations['videos']:
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            # Wanted to use video_start/end_frame, but there seems to be a bug with metadata in Ego4D data so
            # that length of clips would be zero. Thus, calc frames from seconds.
            # 30 fps is safe to assume for canonical videos
            start = int(clip['video_start_sec'] * 30)
            end = int(clip['video_end_sec'] * 30)
            assert start != end, f'{start}, {end}, {clip_uid}'
            work_items.append(WorkItem(
                video['video_uid'], clip_uid,
                start, end
            ))
    return work_items


def main(nlq_file: Path, video_features_dir: Path, output_dir: Path,
         feature_window_stride=16, num_workers=16):
    annotations = json.loads(nlq_file.read_text())
    all_work_items = _extract_work_items(annotations)
    all_work_items = [w for w in all_work_items if not w.output_file(output_dir).is_file()]
    print(f'Will extract {len(all_work_items)} clip features...')

    with Pool(num_workers) as pool, Manager() as manager:
        queue = manager.Queue()
        pool.map_async(partial(WorkItem.do_work,
                               video_features_dir=video_features_dir,
                               output_dir=output_dir,
                               feature_window_stride=feature_window_stride,
                               progress_queue=queue),
                       all_work_items)
        with tqdm(total=len(all_work_items)) as pbar:
            while pbar.n < len(all_work_items):
                pbar.update(queue.get(block=True))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--annotation_file', type=existing_path, required=True,
                        help='Ego4D Annotation JSON file containing annotations for which to extract clip features. '
                             'Should contain "videos" array, where each item has a "clips" array '
                             '(e.g. NLQ annotations).')
    parser.add_argument('--video_features_dir', type=existing_path, required=True,
                        help='Directory where to find pre-extracted Ego4D video features.')
    parser.add_argument('--output_dir', type=existing_path, required=True,
                        help='Directory where to place output files. They are named after the clip_uid.')
    parser.add_argument('--feature_window_stride', type=int, default=16,
                        help='Stride of window used to produce the features in video_features_dir')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='Number of parallel workers')

    args = parser.parse_args()
    main(args.annotation_file, args.video_features_dir, args.output_dir,
         args.feature_window_stride, args.num_workers)


if __name__ == '__main__':
    cli_main()
