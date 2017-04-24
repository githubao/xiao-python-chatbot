#!/usr/bin/env python
# encoding: utf-8

"""
@description: 入门模型

@author: BaoQiang
@time: 2017/4/24 21:14
"""

import tensorflow as tf
import pandas as pd
import numpy as np

test_df = None


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
    y_random = [predict_random(test_df.Context[x], test_df.iloc[x, 1:].valus) for x in range(len(test_df))]
    y_test = np.zeros(len(y_random))
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y_random, y_test), n))


def main():
    predict1()


if __name__ == '__main__':
    main()
