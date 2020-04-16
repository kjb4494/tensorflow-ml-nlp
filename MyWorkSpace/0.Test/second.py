import os
import tensorflow as tf
from tensorflow.keras import preprocessing

samples = ['너 오늘 이뻐 보인다',
           '나는 오늘 기분이 더러워',
           '끝내주는데, 좋은 일이 있나봐',
           '나 좋은 일이 생겼어',
           '아 오늘 진짜 짜증나',
           '환상적인데, 정말 좋은거 같아']

label = [[1], [0], [1], [1], [0], [1]]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index

EPOCH = 100

VOCAB_SIZE = len(word_index) + 1
EMB_SIZE = 128

DATA_OUT_PATH = './data_out/'


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((sequences, label))
    dataset = dataset.repeat(EPOCH)
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(len(sequences))
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    embed_input = tf.keras.layers.Embedding(VOCAB_SIZE, EMB_SIZE)(features)
    embed_input = tf.reduce_mean(embed_input, axis=1)

    hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(embed_input)
    output_layer = tf.keras.layers.Dense(1)(hidden_layer)
    output = tf.nn.sigmoid(output_layer)

    loss = tf.losses.mean_squared_error(output, labels)

    if TRAIN:
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            loss=loss
        )


if __name__ == '__main__':
    if not os.path.exists(DATA_OUT_PATH):
        os.makedirs(DATA_OUT_PATH)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=DATA_OUT_PATH+'checkpoint/dnn')
    estimator.train(train_input_fn)
