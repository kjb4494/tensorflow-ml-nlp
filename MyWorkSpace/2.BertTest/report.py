import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import seaborn as sns
import pandas as pd
import pickle
import time

from main import get_feature_map, get_bert_finetuning_model

test = pd.read_table("nsmc/" + "ratings_test.txt", nrows=2000)
with open('./output/bertembedding.pkl', 'rb') as f:
    bert_embedded = pickle.load(f)


def get_tsne_plot(rot1=-20, rot2=100):
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    colors = 'b', 'r'
    labels = 0, 1
    for i, c, label in zip(range(np.shape(bert_embedded)[0]), colors, labels):
        ax.scatter(
            bert_embedded[test['label'] == label, 0],
            bert_embedded[test['label'] == label, 1],
            bert_embedded[test['label'] == label, 2],
            s=2, c=c, alpha=0.5
        )
    ax.view_init(rot1, rot2)
    print("rot1:%d" % rot1, "rot2:%d" % rot2)
    plt.legend(labels, loc='upper right')
    plt.show()


def main():
    for j in range(-180, 180, 45):
        for i in range(-180, 180, 45):
            get_tsne_plot(i, j)


if __name__ == '__main__':
    main()
