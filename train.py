#!/usr/bin/env python
# encoding: utf-8

import model
import numpy as np
import tensorflow as tf

from graph import Graph
from loader import decoder
from params import params
from params import dump
from path import Path

class Train(object):
    def __init__(self, x, y_):
        cut = int(x.shape[0] * params.test_partial)
        x_tr, y_tr = x[:cut, :], y_[:cut, :]
        x_te, y_te = x[cut:, :], y_[cut:, :]
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        pardim = x.shape[1]
        if params.add_noise:
            pardim += 1
        tardim = y_.shape[1]
        self.graph = Graph(pardim, tardim)
        self.sess_init()

    def __del__(self):
        if self.sess:
            self.sess.close()

    def train(self, steps = params.train_steps):
        outpd = Path(params.outp_dir)
        sess = self.sess
        for step in range(steps):
            self._trainer(sess)
            if step % 100 == 0:
                R, g, s= self._analysiser(sess)
                print("Steps", g,": RMSE", R)
                self.saver.save(sess, outpd/"model",
                           global_step=g)
                self.writer.add_summary(s, g)

    def sess_init(self):
        outpd = Path(params.outp_dir)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(
            outpd, self.sess.graph)
        ckpt = tf.train.get_checkpoint_state(outpd)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)


    def _trainer(self, sess):
        x, y_ = self.sample(test=False)
        sess.run([self.graph.train_op],
                 feed_dict=self.graph.feed_dict(x, y_, False))

    def _analysiser(self, sess):
        x, y_ = self.sample(test=True)
        (R,A,Z,logZ), g, sum = sess.run([self.graph.sum, self.graph.g, self.graph.analysis],
                        feed_dict=self.graph.feed_dict(x, y_, True))
        dump(Path(params.outp_dir)/"RAZ.txt",
             {
                 "R": float(R),
                 "A": float(A),
                 "Z": float(Z),
                 "logZ": float(logZ),
                 "global_steps": int(g)
             })
        return R,g, sum

    def sample(self, test=False):
        if not test:
            xs, ys = model.sampling(self.x_tr, self.y_tr, params.batch_size)
            nxs = model.add_noise(xs) if params.add_noise else xs
        else:
            xs, ys = self.x_te, self.y_te
            nxs = model.add_noise(xs, zeros=True) if params.add_noise else xs
        return nxs, ys


    def predict(self, xargs):
        if (xargs > 1).any() or (xargs < -1).any():
            return np.nan
        decoded = decoder(xargs)
        return float(self.sess.run(self.graph.y, feed_dict=self.graph.feed_dict(decoded, np.ones((1,1)), True)))

from loader import load
def main():
    t = Train(*load())
    t.train()

if __name__ == "__main__":
    if params.multisets:
        for s in params.multisets:
            params.apply_set(s)
            main()
    else:
        main()







