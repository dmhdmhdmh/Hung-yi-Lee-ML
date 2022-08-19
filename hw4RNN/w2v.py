# w2v.py
# 這個block是用來訓練word to vector 的 word embedding
# 注意！這個block在訓練word to vector時是用cpu，可能要花到10分鐘以上
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec
from utils import *

def train_word2vec(x):
    # 訓練word to vector 的 word embedding
    model = word2vec.Word2Vec(x, vector_size=250, window=5, min_count=5, workers=12, epochs=10, sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('D:/homeworkpy/Data/hw4/training_label.txt')
    train_x_no_label = load_training_data('D:/homeworkpy/Data/hw4/training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('D:/homeworkpy/Data/hw4/testing_data.txt')

    model = train_word2vec(train_x + train_x_no_label + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join('D:/homeworkpy/hw4RNN', 'w2v_all.model'))