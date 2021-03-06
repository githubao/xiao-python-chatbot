#!/usr/bin/env python
# encoding: utf-8

"""
@description: 双端编码器

@author: BaoQiang
@time: 2017/4/25 20:58
"""

import tensorflow as tf
import numpy as np
from xiaochat.model import helpers


def get_embeddings(hparams):
    if hparams.glove_path and hparams.vocab_path:
        tf.logging.info('Loading Glove embeddings')
        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             hparams.embedding_dim)
    else:
        tf.logging.info('No glove/vocab path specified, starting with random embeddings.')
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable(
        'word_embeddings',
        shape=[hparams.vocab_size, hparams.embedding_dim],
        initializer=initializer
    )


def dual_encoder_model(hparams, mode, context, context_len, utterance, utterance_len, targets):
    embeddings_W = get_embeddings(hparams)
    context_embedded = tf.nn.embedding_lookup(
        embeddings_W, context, name='embed_context')
    utterance_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterance, name='embed_utterance')

    with tf.variable_scope('rnn') as vs:
        cell = tf.contrib.rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, tf.concat([context_embedded, utterance_embedded], axis=0),
                                                    sequence_length=tf.concat([context_len, utterance_len], axis=0),
                                                    dtype=tf.float32)
        encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, axis=0)

    with tf.variable_scope('prediction') as vs:
        M = tf.get_variable('M', shape=[hparams.rnn_dim, hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())

        generated_response = tf.matmul(encoding_context, M)
        generated_response = tf.expand_dims(generated_response, 2)
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)

        logits = tf.matmul(generated_response, encoding_utterance, True)
        logits = tf.squeeze(logits, [2])

        probs = tf.sigmoid(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    return probs, mean_loss
