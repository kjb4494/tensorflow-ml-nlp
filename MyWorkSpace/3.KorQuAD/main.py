import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import keras

from keras.models import load_model
from keras import backend as K
from keras import Input, Model
from keras import optimizers
from keras.layers import Layer
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps
from keras_radam import RAdam

import os
import re
import shutil
import json
import warnings
import random
from tqdm import tqdm

import korquad
from tokenizer import get_token_dict, get_reverse_token_dict, InheritTokenizer
from data_converter import KorQuADDataConverter, KorQuADPredDataConverter
from layer import NonMasking, LayerStart, LayerEnd

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

SEQ_LEN = 384
BATCH_SIZE = 10
EPOCHS = 1
LR = 3e-5

pretrained_path = 'multi_cased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


def get_bert_finetunning_model(model):
    inputs = model.inputs[:2]
    dense = model.output
    x = NonMasking()(dense)
    outputs_start = LayerStart(SEQ_LEN)(x)
    outputs_end = LayerEnd(SEQ_LEN)(x)
    bert_model = keras.models.Model(inputs, [outputs_start, outputs_end])
    bert_model.compile(
        optimizer=RAdam(learning_rate=LR, decay=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return bert_model


def main_train():
    train = korquad.get_data_frame('./input/KorQuAD_v1.0_train.json')
    tokenizer = InheritTokenizer(get_token_dict(vocab_path=vocab_path))

    # 토큰화 테스트
    # print(tokenizer.tokenize("한국어 토큰화."), tokenizer.tokenize("잘 되니?"))

    # 역키값 토큰 사전
    # print(get_reverse_token_dict(get_token_dict(vocab_path=vocab_path)))

    korquad_data_converter = KorQuADDataConverter(tokenizer=tokenizer, pandas_data_frame=train, seq_len=SEQ_LEN)
    train_x, train_y = korquad_data_converter.load_data()

    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=False,
        trainable=True,
        seq_len=SEQ_LEN
    )

    # 모델 요약 리포트
    # model.summary()

    sess = K.get_session()
    uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
    init = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
    sess.run(init)

    # 1차 훈련
    bert_model = get_bert_finetunning_model(model)
    bert_model.fit(train_x, train_y, batch_size=BATCH_SIZE, validation_split=0.05, shuffle=False, verbose=1)
    bert_model.save_weights('./output/korquad_wordpiece_1.h5')

    # 2차 훈련
    bert_model.compile(optimizer=RAdam(learning_rate=0.00003, decay=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    bert_model.fit(train_x, train_y, batch_size=BATCH_SIZE, shuffle=False, verbose=1)
    bert_model.save_weights('./output/korquad_wordpiece_2.h5')

    # 3차 훈련
    bert_model.compile(optimizer=RAdam(learning_rate=0.00001, decay=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    bert_model.save_weights('./output/korquad_wordpiece_3.h5')


def predict_letter(bert_model, tokenizer, question, doc, real_answer):

    korquad_pred_data_converter = KorQuADPredDataConverter(
        doc=doc,
        question=question,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN
    )
    test_input = korquad_pred_data_converter.load_data()
    test_start, test_end = bert_model.predict(test_input)
    indexes = tokenizer.encode(question, doc, max_len=SEQ_LEN)[0]
    start = np.argmax(test_start, axis=1).item()
    end = np.argmax(test_end, axis=1).item()

    print("Question: {}".format(question))
    print("Context: {}".format(doc))

    sentences = []
    reverse_token_dict = get_reverse_token_dict(get_token_dict(vocab_path=vocab_path))
    for i in range(start, end + 1):
        token_based_word = reverse_token_dict[indexes[i]]
        sentences.append(token_based_word)
    print("Pred Answer: {}".format(sentences))

    for i, token in enumerate(sentences):
        sentences[i] = token.replace('##', '') if '##' in token else ' ' + token
    print("Untokenized Answer: {}".format(''.join(sentences)))

    print("Real Answer: {}".format(real_answer))

    print("= = " * 30)


# 훈련 시간이 너무 오래걸려서 프리트레이닝 정보가 있는 파일을 로드하여 테스트
def main_test():
    tokenizer = InheritTokenizer(get_token_dict(vocab_path=vocab_path))
    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=False,
        trainable=True,
        seq_len=SEQ_LEN
    )
    bert_model = get_bert_finetunning_model(model=model)
    bert_model.load_weights('./output/korquad_wordpiece_3.h5')
    dev = korquad.get_data_frame('./input/KorQuAD_v1.0_dev.json')
    for i in random.sample(range(1000), 1000):
        doc = dev['context'][i]
        question = dev['question'][i]
        answer = dev['text'][i]
        predict_letter(
            bert_model=bert_model,
            tokenizer=tokenizer,
            question=question,
            doc=doc,
            real_answer=answer
        )


if __name__ == '__main__':
    # main_train()
    main_test()
