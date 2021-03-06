#!/usr/bin/env python
# encoding: utf-8

"""
@description:  产生评估矩阵

@author: BaoQiang
@time: 2017/4/25 20:40
"""

import tensorflow as tf
import functools
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics['recall_at_{}'.format(k)] = MetricSpec(metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k, k=k))
    return eval_metrics
