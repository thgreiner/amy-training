import lzma
import pickle
from random import randint

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from chess import Board

from amy.chess.representation import Repr2D
from amy.find_train_files import find_train_files
from amy.network import load_or_create_model

SAMPLE = 1


def quantize_model(model_name: str):
    with tfmot.quantization.keras.quantize_scope():
        model = load_or_create_model(model_name)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(model_name.replace("h5", "tflite"), "wb") as fout:
        fout.write(tflite_quant_model)


def _representative_dataset_gen():
    repr2d = Repr2D()
    yield [repr2d.board_to_array(Board()).reshape(1, 8, 8, 19).astype("float32")]

    files = find_train_files(60_000, 1, True)
    cnt = 0

    for filename in files:
        print(filename)
        with lzma.open(filename, "rb") as fin:
            try:
                while cnt < 200:
                    item = pickle.load(fin)
                    if randint(0, 999) < SAMPLE:
                        features = item.data_board.reshape(1, 8, 8, 19).astype(
                            "float32"
                        )
                        yield [features]
                        cnt += 1
                        print(cnt, end="\r")
            except EOFError:
                pass

    print(f"Provided {cnt} samples.")
