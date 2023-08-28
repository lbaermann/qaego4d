# <span style="font-variant:small-caps;">QaEgo4D</span> — Episodic-Memory-Based Question Answering on Egocentric Videos

This repository contains the code to reproduce the results of the paper "Where did I leave my keys? —
Episodic-Memory-Based Question Answering on Egocentric Videos". See our
[paper](https://openaccess.thecvf.com/content/CVPR2022W/Ego4D-EPIC/papers/Barmann_Where_Did_I_Leave_My_Keys_-_Episodic-Memory-Based_Question_Answering_CVPRW_2022_paper.pdf)
for more details.

## Abstract

Humans have a remarkable ability to organize, compress and retrieve episodic memories throughout their daily life.
Current AI systems, however, lack comparable capabilities as they are mostly constrained to an analysis with access to
the raw input sequence, assuming an unlimited amount of data storage which is not feasible in realistic deployment
scenarios. For instance, existing Video Question Answering (VideoQA) models typically reason over the video while
already being aware of the question, thus requiring to store the complete video in case the question is not known in
advance.

In this paper, we address this challenge with three main contributions:
First, we propose the Episodic Memory Question Answering (EMQA) task as a specialization of VideoQA. Specifically, EMQA
models are constrained to keep only a constant-sized representation of the video input, thus automatically limiting the
computation requirements at query time. Second, we introduce a new egocentric VideoQA dataset
called <span style="font-variant:small-caps;">QaEgo4D</span>. It is the by far largest egocentric VideoQA dataset and
video length is unprecedented in VideoQA datasets in general. Third, we present extensive experiments on the new
dataset, comparing various baselines models in both the VideoQA as well as the EMQA setting. To facilitate future
research on egocentric VideoQA as well as episodic memory representation and retrieval, we publish our code and dataset.

## Using the dataset

To use the <span style="font-variant:small-caps;">QaEgo4D</span> dataset introduced in our paper, please follow these
steps:

1. <span style="font-variant:small-caps;">QaEgo4D</span> builds on the Ego4D v1 videos and annotations. If you do not have
   access to Ego4D already, you should follow the steps at the [Ego4D website](https://ego4d-data.org/docs/start-here/)
2. To get access to <span style="font-variant:small-caps;">QaEgo4D</span>, please fill out
   this [Google form](https://forms.gle/Gxs93wwC5YYJtjqh8). You will need to sign a license agreement, but there are no
   fees if you use the data for non-commercial research purposes.
3. Download the Ego4D annotations and NLQ clips if you have not done so already. See
   the [Ego4D website](https://ego4d-data.org/docs/start-here/)
4. After you have access to both Ego4D and <span style="font-variant:small-caps;">QaEgo4D</span>, you can generate
   self-contained VideoQA annotation files
   using `python3 tools/create_pure_videoqa_json.py --ego4d /path/to/ego4d --qaego4d /path/to/qaego4d/answers.json`.
   `/path/to/ego4d` is the directory where you placed the Ego4D download, containing
   the `v1/annotations/nlq_{train,val}.json` files. This produces `/path/to/qaego4d/annotations.{train,val,test}.json`.

The `annotations.*.json` files are JSON arrays, where each object has the following structure:
```
{
   "video_id": "abcdef00-0000-0000-0000-123456789abc", 
   "sample_id": "12345678-1234-1234-1234-123456789abc_3", 
   "question": "Where did I leave my keys?", 
   "answer": "on the table", 
   "moment_start_frame": 42, 
   "moment_end_frame": 53
}
```

## Code

In order to reproduce the experiments, prepare your workspace:

1. Follow the instructions above to get the dataset and features.
2. Create a conda / python virtual environment (Python 3.9.7)
3. Install the requirements in `requirements.txt`
4. Prepare the features:
   1. Download the pre-extracted [Ego4D features](https://ego4d-data.org/docs/data/features/) if you have not done so
      already.
   2. Ego4D features are provided for each canonical video, while the NLQ task and thus also VideoQA works on the
      canonical clips. To extract features for each clip,
      use `python tools/extract_ego4d_clip_features.py --annotation_file /path/to/ego4d/v1/annotations/nlq_train.json --video_features_dir /path/to/ego4d/v1/slowfast8x8_r101_k400 --output_dir /choose/your/clip_feature_dir`
      and do the same again with `nlq_val.json`
   3. Aggregate the features into a single file
      using `python tools/aggregate_features_to_hdf5.py /choose/your/clip_feature_dir`. This
      produces `slowfast8x8_r101_k400.hdf5` in the current working directory.
5. Place or link the <span style="font-variant:small-caps;">QaEgo4D</span> data (`annotations.*.json`
   and `slowfast8x8_r101_k400.hdf5`) into `datasets/ego4d`.

To run an experiment, use `bash experiment/run.sh`. All configuration files can be found in the `config` dir.


## Cite
```
@InProceedings{Baermann_2022_CVPR,
    author    = {B\"armann, Leonard and Waibel, Alex},
    title     = {Where Did I Leave My Keys? - Episodic-Memory-Based Question Answering on Egocentric Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1560-1568}
}
```
