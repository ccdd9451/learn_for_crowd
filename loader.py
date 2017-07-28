#!/usr/bin/env python
# encoding: utf-8

import pickle
import numpy as np

from params import params
from path import Path

dataf = Path(params.data_dir)

with open(dataf, "rb") as f:
    datas = pickle.load(f)

def load():
    col_names = [col_name for col_name in datas.columns if
                     col_name.endswith("th_obstacle") or
                     col_name.endswith("th_region") or
                     col_name.endswith("th_ai") ]
    x_pre = datas[col_names]
    x_pre_col = x_pre.columns[(x_pre - x_pre.mean() > 0.01).any()]
    x = x_pre[x_pre_col].as_matrix()
    y_ = datas[params.out_cols].as_matrix()
    return x, y_

envs_cols = [col_name for col_name in datas.columns if
                     col_name.endswith("th_obstacle")]
envs = datas[envs_cols].as_matrix()
envmax = envs.max(0)
envmin = envs.min(0)
envavg = (envmax + envmin) / 2
envln = (envmax - envmin) / 2
el = int(envs.shape[1])

cols = [col_name for col_name in datas.columns if
                     col_name.endswith("th_region") or
                     col_name.endswith("th_ai") ]
otherdat = datas[cols].as_matrix().mean(0)

def decoder(xargs):
    obstacle = np.add(np.multiply(xargs, envln), envavg)
    if params.add_noise:
       decoded = np.concatenate((obstacle, otherdat, [0]), axis=0)
    else:
       decoded = np.concatenate((obstacle, otherdat), axis=0)
    return decoded.reshape(1, len(decoded))


def encoder(xargs):
    return np.divide(np.subtract(xargs, envavg), envln)

