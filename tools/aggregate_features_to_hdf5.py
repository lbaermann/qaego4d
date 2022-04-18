import sys
from pathlib import Path

import h5py
import torch
from tqdm import tqdm


def main():
    features_dir = Path(sys.argv[1])
    all_feature_files = list(features_dir.glob('*.pt'))

    with h5py.File('slowfast8x8_r101_k400.hdf5', 'a') as h5_file, tqdm(all_feature_files) as iterator:
        features_file: Path
        for features_file in iterator:
            try:
                clip_uid = features_file.name.replace('.pt', '')
                if clip_uid in h5_file:
                    continue
                features = torch.load(str(features_file))
                h5_file.create_dataset(clip_uid, data=features, compression="gzip")
            except BaseException as e:
                raise RuntimeError(f'Error during {features_file}: {e}')


if __name__ == '__main__':
    main()
