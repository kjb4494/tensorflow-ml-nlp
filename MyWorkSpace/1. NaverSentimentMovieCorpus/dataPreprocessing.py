import numpy as np
import pandas as pd
import re
import json
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

DATA_IN_PATH = './data_in/'
TRAIN_INPUT_DATA = 'nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'
TEST_INPUT_DATA = 'nsmc_test_input.npy'
TEST_LABEL_DATA = 'nsmc_test_label.npy'
DATA_CONFIGS = 'data_configs.json'

train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train.txt', header=0, delimiter='\t', quoting=3)
test_data = pd.read_csv(DATA_IN_PATH + 'ratings_test.txt', header=0, delimiter='\t', quoting=3)
okt = Okt()
tokenizer = Tokenizer()
stop_words = []

with open('./reference_data/stop_word.txt') as file:
    for line in file.readlines():
        stop_words.append(line.strip())


def test_01():
    print(train_data['document'][:5])


def test_02():
    review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', "", train_data['document'][0])
    print(review_text)

    # okt 객체를 이용해 문장을 형태소 단위로 나눔
    review_text = okt.morphs(review_text, stem=True)
    print(review_text)

    # 불용어를 제거함
    clean_review = [word for word in review_text if word not in stop_words]
    print(clean_review)


def pre_processing(review, remove_stopwords = False):
    review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', "", review)
    word_review = okt.morphs(review_text, stem=True)

    if remove_stopwords:
        word_review = [word for word in word_review if word not in stop_words]
    return word_review


def extract_train_data():
    clean_train_review = []
    print('단어 데이터 추출 시작')
    for i, review in enumerate(train_data['document']):
        if type(review) == str:
            clean_train_review.append(pre_processing(review, remove_stopwords=True))
        else:
            clean_train_review.append([])
        print('\r진행률: {:.2f}%'.format(i/len(train_data)*100), end='')
    print('\n단어 데이터 추출 완료')

    print('전처리 및 저장 프로세스 시작')
    tokenizer.fit_on_texts(clean_train_review)
    train_sequences = tokenizer.texts_to_sequences(clean_train_review)

    train_inputs = pad_sequences(train_sequences, maxlen=8, padding='post')
    train_labels = np.array(train_data['label'])

    # 전처리된 학습 데이터 저장
    np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
    np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)

    print('훈련 데이터 전처리 완료')


def extract_test_data():
    clean_test_review = []
    print('단어 데이터 추출 시작')
    for i, review in enumerate(test_data['document']):
        if type(review) == str:
            clean_test_review.append(pre_processing(review, remove_stopwords=True))
        else:
            clean_test_review.append([])
        print('\r진행률: {:.2f}%'.format(i/len(test_data)*100), end='')
    print('\n단어 데이터 추출 완료')

    print('전처리 공정 시작')
    tokenizer.fit_on_texts(clean_test_review)
    test_sequences = tokenizer.texts_to_sequences(clean_test_review)

    test_inputs = pad_sequences(test_sequences, maxlen=8, padding='post')
    test_labels = np.array(test_data['label'])

    np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
    np.save(open(DATA_IN_PATH + TEST_LABEL_DATA, 'wb'), test_labels)
    print('테스트 데이터 전처리 완료')


def extract_main():
    extract_train_data()
    extract_test_data()

    print('단어 사전 생성 시작')
    word_vocab = tokenizer.word_index
    data_configs = {'vocab': word_vocab, 'vocab_size': len(word_vocab) + 1}
    json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)
    print('단어 사전 생성 완료')


if __name__ == '__main__':
    # test_01()
    # test_02()
    extract_main()
