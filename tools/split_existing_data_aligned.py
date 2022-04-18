import json
import random
import sys
from pathlib import Path

input_data_file = Path(sys.argv[1])
clip_to_video_file = Path(sys.argv[2])
input_samples = json.loads(input_data_file.read_text())
clip_to_video = json.loads(clip_to_video_file.read_text())

all_clips = {s['video_id'] for s in input_samples}
all_videos = {clip_to_video[clip] for clip in all_clips}

frac_val = 0.5
assert frac_val < 1

num_val = round(len(all_videos) * frac_val)
num_test = len(all_videos) - num_val
assert num_test >= 1

val_videos = set(random.sample(sorted(all_videos), num_val))
test_videos = all_videos - val_videos

val_clips = {clip for clip in all_clips if clip_to_video[clip] in val_videos}
test_clips = {clip for clip in all_clips if clip_to_video[clip] in test_videos}

val_samples = [s for s in input_samples if s['video_id'] in val_clips]
test_samples = [s for s in input_samples if s['video_id'] in test_clips]

print(f'Splits: videos/clips/samples ')
print(f'Val   : {len(val_videos)}/{len(val_clips)}/{len(val_samples)}')
print(f'Test  : {len(test_videos)}/{len(test_clips)}/{len(test_samples)}')

Path('pure_emqa_val.json').write_text(json.dumps(val_samples))
Path('pure_emqa_test.json').write_text(json.dumps(test_samples))

Path('split_videos.json').write_text(json.dumps({'val': sorted(val_videos), 'test': sorted(test_videos)}))
Path('split_clips.json').write_text(json.dumps({'val': sorted(val_clips), 'test': sorted(test_clips)}))
