#!/usr/bin/env python
# encoding: utf-8

import model
import tensorflow as tf

from params import params



class Graph(object):
    def __init__(self, pardim, tardim):
        tf.reset_default_graph()
        self.nodes = []
        self.p_dict = {}
        all_nodes = [pardim] + params.node_list + [tardim]
        self._create_graph(all_nodes)

    def _create_graph(self, nodelist):
        pardim = nodelist[0]
        self.x = tf.placeholder(tf.float32,
                               [None, pardim],
                               name="x")
        self.nodes.append(self.x)

        for node in nodelist[1:-1]:
            if type(node) is int:
                new = model.nn_layer(self.nodes[-1],
                                     node)
            elif node is "d": # dropout
                new, keep_prob = model.dropout(self.nodes[-1])
                self.p_dict[keep_prob] = (params.keep_prob, 1)

            self.nodes.append(new)
        new = model.nn_layer(self.nodes[-1], nodelist[-1],
                             act=tf.identity)
        self.nodes.append(new)

        self.y = self.nodes[-1]
        tardim = self.nodes[-1].get_shape()[1]
        self.ref_y = tf.placeholder(tf.float32,
                                   [None, tardim],
                                   name="y")

        self.R, self.sum, _ = model.analysis(self.y, self.ref_y)
        self.analysis = tf.summary.merge_all()
        self.g = tf.Variable(0, name="global_step")

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(
                params.learning_rate
            ).minimize(self.R, global_step=self.g)

    def feed_dict(self, x, y_, test=False):
        d = {}
        index = 1 if test else 0
        for key, content in self.p_dict.items():
            d[key] = content[index]
        d[self.x] = x
        d[self.ref_y] = y_
        return d





