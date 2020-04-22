import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import seaborn as sns

# from wordcloud import WordCloud

DATA_IN_PATH = './data_in/'
train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train.txt', header=0, delimiter='\t', quoting=3)


def get_data_size():
    print('파일 크기: ')
    for file in os.listdir(DATA_IN_PATH):
        if 'txt' in file:
            print("{}{}MB".format(file.ljust(30), round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)))


def get_train_data():
    print(train_data.head())
    print('전체 학습 데이터의 개수: {}'.format(len(train_data)))


# 훈련 데이터의 각 리뷰 길이 추출
def get_train_data_review_length():
    train_length = train_data['document'].astype(str).apply(len)
    print(train_length.head())

    plt.figure(figsize=(12, 5))
    plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of length of review')
    plt.xlabel('Length of review')
    plt.ylabel('Number of review')

    plt.show()

    print('리뷰 길이 최댓값: {}'.format(np.max(train_length)))
    print('리뷰 길이 최솟값: {}'.format(np.min(train_length)))
    print('리뷰 길이 평균값: {:.2f}'.format(np.mean(train_length)))
    print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
    print('리뷰 길이 중간값: {}'.format(np.median(train_length)))


def pn_classification():
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(train_data['label'])
    plt.show()

    print('긍정 리뷰 개수: {}'.format(train_data['label'].value_counts()[1]))
    print('부정 리뷰 개수: {}'.format(train_data['label'].value_counts()[0]))


def get_train_data_word_count():
    train_word_counts = train_data['document'].astype(str).apply(lambda x: len(x.split(' ')))

    plt.figure(figsize=(15, 10))
    plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
    plt.title('Log-Histogram of word count in review', fontsize=15)
    plt.yscale('log', nonposy='clip')
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Number of reviews', fontsize=15)
    plt.show()

    print('리뷰 단어 개수 최댓값: {}'.format(np.max(train_word_counts)))
    print('리뷰 단어 개수 최솟값: {}'.format(np.min(train_word_counts)))
    print('리뷰 단어 평균값: {:.2f}'.format(np.mean(train_word_counts)))
    print('리뷰 단어 표준편차: {:.2f}'.format(np.std(train_word_counts)))
    print('리뷰 단어 중간값: {}'.format(np.median(train_word_counts)))


def special_characters_analysis():
    qmarks = np.mean(train_data['document'].astype(str).apply(lambda x: '?' in x))
    fullstop = np.mean(train_data['document'].astype(str).apply(lambda x: '.' in x))

    print('물음표가 있는 질문: {:.2f}%'.format(qmarks * 100))
    print('마침표가 있는 질문: {:.2f}%'.format(fullstop * 100))


if __name__ == '__main__':
    # get_data_size()
    # get_train_data()
    # get_train_data_review_length()
    # pn_classification()
    # get_train_data_word_count()
    special_characters_analysis()
