import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split

DATA_IN_PATH = './data_in'
DATA_OUT_PATH = './data_out'
INPUT_TRAIN_DATA_FILE_NAME = 'nsmc_train_input.npy'
LABEL_TRAIN_DATA_FILE_NAME = 'nsmc_train_label_npy'
DATA_CONFIGS_FILE_NAME = 'data_configs.json'

input_data = np.load(open(DATA_IN_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'rb'))
label_data = np.load(open(DATA_IN_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS_FILE_NAME, 'r'))

TEST_SPLIT = 0.1
RNG_SEED = 13371447
BATCH_SIZE = 16
NUM_EPOCHS = 5

input_train, input_eval, label_train, label_eval = train_test_split(
    input_data, label_data, test_size=TEST_SPLIT, random_state=RNG_SEED
)


def mapping_fn(x, y):
    input_x, label = {'x': x}, y
    return input_x, label


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.shuffle(buffer_size=len(input_train))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_eval, label_eval))
    dataset = dataset.shuffle(buffer_size=len(input_eval))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

