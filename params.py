#!/usr/bin/env python
# encoding: utf-8

import yaml
from pathlib import Path

class Param(dict):
    def __init__(self):
        self.__dict__ = self
        scdir = Path(__file__).with_name("feed_dict")
        self.update(load(scdir)["feed_dict"])
        self.update(sets_contents[
            load(scdir)["default_set"]])
        cwdir = Path(".")/"feed_dict"
        if cwdir.exists():
            self.update(load(cwdir)["feed_dict"])
            self.update(sets_contents[
                load(cwdir)["default_set"]])
    def apply_set(self, setnum):
        self.update(sets_contents[setnum])

def load(path):
    with open(path, "r") as f:
        return yaml.load(f)

def dump(path, content):
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=None)

if __name__ == "__main__":
    default_dict = {
        "bias_constant": 0.1,
        "weight_stddev": 0.04,
        "keep_prob": 0.8,
        "node_list": [50, "d"],
        "train_steps": 20000,
        "data_dir": "data.file",
        "test_partial": 0.8,
        "add_noise": True,
        "noise_amp": 1,
        "learning_rate": 0.001,
        "batch_size": 2000,
        "multisets": []
    }

    dump("feed_dict", {"feed_dict":default_dict, "default_set": 0})

sets_contents = [ {
        "outp_dir" : "./atime/",
        "out_cols": ["time_avg"]
    }, {
        "outp_dir" : "./acls/",
        "out_cols": ["cls_avg"]
    }, {
        "outp_dir" : "./alen/",
        "out_cols": ["len_avg"]
    }, {
        "outp_dir" : "./aple/",
        "out_cols": ["ple_avg"]
    } ]

params = Param()
