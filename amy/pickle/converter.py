import os
import pickle
import random
from functools import partial
from os import path
from queue import PriorityQueue
from threading import Thread

from amy.pgn.reader import pos_generator


def convert_pgn_to_pickle(file_name: str, output_dir: str, nfiles: int, split: int):
    """Convert a PGN file to a pickle training file."""
    queue = PriorityQueue()

    pos_gen = partial(pos_generator, file_name, True, queue)

    t = Thread(target=pos_gen)
    t.start()

    if not path.exists(output_dir):
        os.mkdir(output_dir)
    validation_file = open(path.join(output_dir, "validation.pkl"), "wb")

    train_files = []
    for i in range(nfiles):
        train_files.append(open(path.join(output_dir, f"train-{i:02d}.pkl"), "wb"))

    val_cnt = 0
    train_cnt = 0

    while train_cnt < 4_000_000:

        item = queue.get()
        if item.data_board is None:
            break

        is_validation = random.randint(0, 99) < split
        file = (
            validation_file
            if is_validation
            else train_files[random.randint(0, nfiles - 1)]
        )
        pickle.dump(item, file)

        if is_validation:
            val_cnt += 1
        else:
            train_cnt += 1

    validation_file.close()
    for f in train_files:
        f.close()

    print(f"Positions: {train_cnt}/{val_cnt} (training/validation)")

    with open(path.join(output_dir, "position_count"), "w") as pfile:
        pfile.write(str(train_cnt))
