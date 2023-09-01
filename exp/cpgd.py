"""
Implementation of constrained adversarial attacks using CPGD from.
This implementation uses a different constraint validation strategy,
and the constraint checking is built into the attack. For details,

see paper: https://arxiv.org/abs/2112.01156
and code: https://github.com/serval-uni-lu/constrained-attacks

The comparison implementation is in comparison/ under its own license.
Minor modifications were applied to make it compatible with more
recent versions of Python and adversarial-robustness-toolkit than
listed in the original distribution.
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from typing import List

from comparison.constraints.relation_constraint import \
    Feature, BaseRelationConstraint
from comparison.constraints.constraints import get_constraints_from_file
from comparison.cpgd.tf_classifier import TensorflowClassifier
from comparison.cpgd.cpgd import CPGD


def get_unsw_constraints() -> List[BaseRelationConstraint]:
    # TODO: implement
    # there must be at least two constraints to compute loss
    # see example in reference/constraints_ex.py
    pass


def get_iot_constraints() -> List[BaseRelationConstraint]:
    # TODO: implement
    # there must be at least two constraints to compute loss
    pass


def init_constraints(feat_file):
    if 'unsw' in feat_file:
        c_set = get_unsw_constraints()
    elif 'iot' in feat_file:
        c_set = get_iot_constraints()
    else:
        c_set = None
    if not c_set or len(c_set) < 2:
        g1 = Feature(0) <= Feature(0)
        g2 = Feature(1) <= Feature(1)
        c_set = [g1, g2]
    return feat_file, c_set


def get_scaler(path: str):
    df = pd.read_csv(path, low_memory=False)
    min_c = df["min"].to_numpy()
    max_c = df["max"].to_numpy()
    _scaler = MinMaxScaler(feature_range=(0, 1))
    _scaler.fit([min_c, max_c])
    return _scaler


def cpgd_apply_predict(
        keras_nn: Sequential, x: np.ndarray, y: np.ndarray,
        enable_constr: bool, feat_file: str, **config
):
    args_ = {**config['args'], 'enable_constraints': enable_constr}
    constraints = get_constraints_from_file(*init_constraints(feat_file))
    scaler = get_scaler(feat_file)
    model = TensorflowClassifier(keras_nn)  # wrap in their interface
    pipe = Pipeline(steps=[("preprocessing", scaler), ("model", model)])
    x_adv = CPGD(pipe, constraints, **args_).generate(x, y)
    x_adv = x_adv.reshape(-1, x_adv.shape[-1])  # remove extra axis
    y_adv = model.predict(x_adv)
    return x_adv, y_adv
