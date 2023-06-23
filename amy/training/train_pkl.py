import lzma
import pickle
import time
from functools import partial
from queue import PriorityQueue
from random import randint
from threading import Thread

import tensorflow_model_optimization as tfmot
from prometheus_client import start_http_server

from amy.find_train_files import find_train_files
from amy.network import load_or_create_model
from amy.pgn.reader import end_of_input_item, randomize_item
from amy.training.loop import train_epoch


def train_from_pkl(model_name: str, batch_size: int, test_mode: bool):
    """Train a model."""
    with tfmot.quantization.keras.quantize_scope():
        model = load_or_create_model(model_name)
    model.summary()

    queue = PriorityQueue()

    for port in range(9099, 9104):
        try:
            start_http_server(port)
            print(f"Started http server on port {port}")
            break
        except OSError:
            pass

    for epoch in range(1):

        _start_pos_gen_thread(queue, test_mode)

        train_epoch(model, batch_size, epoch, queue, test_mode)

        if test_mode:
            break

        if model_name is None:
            model.save("combined-model.h5")
        else:
            model.save(model_name)
            history_name = f"{model_name.removesuffix('.h5')}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.h5"
            model.save(f"model_history/{history_name}")


def _start_pos_gen_thread(queue, test_mode):
    pos_gen = partial(_read_pickle, queue, test_mode)

    t = Thread(target=pos_gen)
    t.start()

    if not test_mode:
        _wait_for_queue_to_fill(queue)


def _read_pickle(queue, test_mode):

    sample = 8
    files = find_train_files(750_000, sample, test_mode)

    for filename in files:
        print(f"Reading {filename}", end="\r")
        with lzma.open(filename, "rb") as fin:
            try:
                while True:
                    item = pickle.load(fin)
                    if randint(0, 99) < sample:
                        queue.put(randomize_item(item))
            except EOFError:
                pass

    queue.put(end_of_input_item())


def _wait_for_queue_to_fill(q):
    old_qsize = None
    for i in range(900):
        time.sleep(1)
        print(f"Waiting for queue to fill, current size is {q.qsize()}     ")
        if q.qsize() > 50000:
            break
        if old_qsize is not None and old_qsize == q.qsize():
            break
        old_qsize = q.qsize()
