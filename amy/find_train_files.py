import glob
import os
import random


def find_train_files(position_count: int, sample_rate: int, test_mode: bool):
    """Scan the "data" directory for training data, and return a shuffled list of training files.

    Args:
        position_count: the desired position count
        sample_rate: the sample rate
        test_mode: a flag indicating whether the caller wants training or validation data

    Returns:
        A shuffled list of file names. Sampling positions from these files with the given
        sample rate will give approximately the desired position count.
    """
    pos_count_files = glob.glob("data/*/position_count")

    total_count = 0

    data_files = []

    for name in sorted(pos_count_files, reverse=True):
        with open(name) as pfile:
            cnt = int(pfile.readline())
        total_count += cnt
        print(f"{name} {cnt} {total_count}")
        dir = os.path.dirname(name)
        print(dir)

        if test_mode:
            data_files.extend(glob.glob(os.path.join(dir, "validation.pkl.xz")))
        else:
            data_files.extend(glob.glob(os.path.join(dir, "train-*.pkl.xz")))

        if total_count * sample_rate / 100 > position_count:
            break

    random.shuffle(data_files)
    return data_files
