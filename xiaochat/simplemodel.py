#!/usr/bin/env python
# encoding: utf-8

"""
@description: 入门模型，随机和tf-idf

@author: BaoQiang
@time: 2017/4/24 21:14
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
validation_df = pd.read_csv('data/valid.csv')


def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct / num_examples


def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)


def predict1():
    '''
    Recall @ (1, 10): 0.103805
    Recall @ (2, 10): 0.20777
    Recall @ (5, 10): 0.504915
    Recall @ (10, 10): 1
    '''
    y_random = [predict_random(test_df.Context[x], test_df.iloc[x, 1:].values) for x in range(len(test_df))]
    y_test = np.zeros(len(y_random))
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y_random, y_test, n)))


class TfIdfPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values, data.Utterance.values))

    def predict(self, context, utterances):
        # convert context and utterrances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # the dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]


def predict2():
    '''
    Recall @ (1, 10): 0.242424
    Recall @ (2, 10): 0.333333
    Recall @ (5, 10): 0.707071
    Recall @ (10, 10): 1
    :return: 
    '''
    pred = TfIdfPredictor()
    pred.train(train_df)
    y = [pred.predict(test_df.Context[x], test_df.iloc[x, 1:].values) for x in range(len(test_df))]
    y_test = np.zeros(len(y))
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y, y_test, n)))


def main():
    # predict1()
    predict2()


if __name__ == '__main__':
    main()
