import os
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import pickle
import keras
import codecs
import shutil
import warnings
from keras.models import load_model
from keras import backend as K
from keras import Input, Model
from keras import optimizers
from numpy.f2py.crackfortran import verbose
from tqdm import tqdm
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import AdamWarmup, calc_train_steps
from keras_radam import RAdam
from IPython.display import SVG
from keras.utils import model_to_dot
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from inherit_tokenizer import InheritTokenizer
from data_converter import NsmcDataConverter, SentenceConverter
from data_predictor import NsmcDataPredictor

# 로그 레벨 설정
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# 트레이닝 설정
SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-5

pretrained_path = 'multi_cased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

train = pd.read_table("nsmc/" + "ratings_train.txt")
test = pd.read_table("nsmc/" + "ratings_test.txt", nrows=2000)
# test = pd.read_table("nsmc/" + "ratings_test.txt")


def get_token_dict():
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token = '##' + token.replace('_', '') if '_' in token else token
            token_dict[token] = len(token_dict)
    return token_dict


def get_bert_finetuning_model(model):
    inputs = model.inputs[:2]
    dense = model.layers[-3].output
    outputs = keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        name='real_output'
    )(dense)
    bert_model = keras.models.Model(inputs, outputs)
    bert_model.compile(
        optimizer=RAdam(learning_rate=0.00001, weight_decay=0.0025),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return bert_model


def get_feature_map(model):
    inputs = model.input
    outputs = model.layers[-2].output
    feature_model = Model(inputs, outputs)
    return feature_model


def main():
    tokenizer = InheritTokenizer(get_token_dict())

    # 네이버 영화 리뷰 전처리
    # nsmc_converter = NsmcDataConverter(tokenizer=tokenizer, pandas_data_frame=train, seq_len=SEQ_LEN)
    # train_x, train_y = nsmc_converter.load_data()
    # test_x, test_y = nsmc_converter.load_data()
    # print(train_x)

    # 문장 전처리
    # sentence_converter = SentenceConverter(tokenizer=tokenizer, seq_len=SEQ_LEN)
    # test_sentence = sentence_converter.load_data(["케라스로 버트 해보기 정말 재밌음", "케라스 쉬워 쉬워"])
    # print(test_sentence)

    # 모델 정의
    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=True,
        trainable=True,
        seq_len=SEQ_LEN
    )

    # 모델 생김새 요약 보고서
    # model.summary()

    # 모델 흐름 시각화 - GraphViz 설치가 요구됨
    # print(model_to_dot(get_bert_finetuning_model(model), dpi=65).create(prog='dot', format='svg'))

    # sess = K.get_session()
    # uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
    # init = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
    # sess.run(init)

    # bert_model = get_bert_finetuning_model(model)
    # history = bert_model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=(test_x, test_y), shuffle=True)
    # bert_model.save_weights('./output/bert.h5')

    bert_model = get_bert_finetuning_model(model)
    bert_model.load_weights('./output/bert.h5')

    # 테스트 데이터 예측
    nsmc_data_predictor = NsmcDataPredictor(tokenizer=tokenizer, data_df=test, seq_len=SEQ_LEN)
    test_set = nsmc_data_predictor.load_data()
    # preds = bert_model.predict(test_set)

    # 예측 스코어 리포트
    # y_true = test['label']
    # print(classification_report(y_true, np.round(preds, 0)))

    bert_feature = get_feature_map(bert_model)
    # print(model_to_dot(bert_feature, dpi=65).create(prog='dot', format='svg'))

    bert_weight_list = bert_feature.predict(test_set)
    bert_embedded = PCA(n_components=256).fit_transform(bert_weight_list)
    bert_embedded = TSNE(n_components=3).fit_transform(bert_embedded)

    with open('./output/bertembedding.pkl', 'wb') as f:
        pickle.dump(bert_embedded, f)


if __name__ == '__main__':
    main()
