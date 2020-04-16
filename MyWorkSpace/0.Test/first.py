import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import preprocessing

if __name__ == '__main__':
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

    print("수치화된 텍스트 데이터: {}".format(sequences))
    print("각 단어의 인덱스: {}".format(word_index))
    print("라벨: {}".format(label))

    dataset = tf.data.Dataset.from_tensor_slices((sequences, label))
    # dataset = dataset.batch(2)
    # dataset = dataset.shuffle(len(sequences))
    EPOCH = 2
    dataset = dataset.repeat(EPOCH)
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next_data))
            except tf.errors.OutOfRangeError:
                break
