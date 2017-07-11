#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf

from params import params


def nn_layer(input_tensor, output_dim, act=tf.nn.relu):
    nn_layer.counter = getattr(nn_layer, 'counter', 0) + 1
    input_dim = int(input_tensor.get_shape()[1])

    with tf.name_scope("layer"+str(nn_layer.counter)):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')

        variable_summaries(biases)
        variable_summaries(weights)
        tf.summary.histogram('pre_activations', preactivate)
        tf.summary.histogram('activations', activations)
        return activations

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=params.weight_stddev)
    return tf.Variable(initial, name="weight")


def bias_variable(shape):
    initial = tf.constant(params.bias_constant, shape=shape)
    return tf.Variable(initial, name="biases")

def variable_summaries(var):
    with tf.name_scope('summaries'):
        tf.summary.histogram('histogram', var)

def dropout(hidden1):
    dropout.counter = getattr(dropout, 'counter', 0) + 1
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        return tf.nn.dropout(hidden1, keep_prob), keep_prob

def analysis(var, ref):
    with tf.name_scope("test_grades"):
        mean = tf.reduce_mean(var)
        refmean = tf.reduce_mean(ref)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(ref - refmean)))
        refstddev = tf.sqrt(tf.reduce_mean(tf.square(var - refmean)))
        A = refstddev / refmean
        Z = refstddev / stddev
        As = tf.summary.scalar('A', A)
        Zs = tf.summary.scalar('Z', Z)
    with tf.name_scope("test_grades"):
        R = tf.sqrt(tf.reduce_mean(tf.square(var - ref)))
        Rs = tf.summary.scalar('R', R)
    return R, (R,A,Z), tf.summary.merge([Rs, As, Zs])


def sampling(x, y, batch_size):
    amount = x.shape[0]
    sample = np.random.choice(amount, size=batch_size, replace=False)
    return x[sample,:], y[sample,:]

def add_noise(source, zeros=False):
    amount = source.shape[0]
    if not zeros:
        noise = np.random.randn(amount, 1) * params.noise_amp
    else:
        noise = np.zeros((amount,1))
    return np.concatenate([source, noise], axis = 1)

