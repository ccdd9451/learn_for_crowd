#!/usr/bin/env python
# encoding: utf-8

import pickle

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
    x = datas[col_names].as_matrix()
    y_ = datas[params.out_cols].as_matrix()
    return x, y_


