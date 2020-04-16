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


if __name__ == '__main__':
    pass
