"""
Implementation of constrained adversarial attacks using CPGD.
This implementation uses a different constraint validation strategy,
and the constraint checking is built into the attack. See:

Paper: https://arxiv.org/abs/2112.01156
Source code: https://github.com/serval-uni-lu/constrained-attacks
Experiments: https://github.com/serval-uni-lu/moeva2-ijcai22-replication

The comparison implementation is in comparison/ under its own license.
Minor modifications were applied to make it compatible with more
recent versions of Python, adversarial-robustness-toolkit than the
original distribution.
"""

from typing import List

import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from comparison.constraints.constraints import get_constraints_from_file
from comparison.constraints.relation_constraint import \
    Feature, BaseRelationConstraint, Constant
from comparison.cpgd.cpgd import CPGD
from comparison.cpgd.tf_classifier import TensorflowClassifier


def get_unsw_constraints() -> List[BaseRelationConstraint]:
    # TODO: implement
    # there must be at least two constraints to compute loss
    # see example in reference/constraints_ex.py
    def apply_const_on_tcp_fields(
            a: Feature, b: Feature, c: Feature, d: Feature, e: Feature,
            f: Feature, g: Feature, h: Feature, i: Feature):
        return (((a != Constant(1))
                 or (b == Constant(255) or c == Constant(255))
                 or (d == Constant(0) and c == Constant(0))
                 or e == Constant(1) or f == Constant(0)
                 or d == Constant(0) or g != Constant(1))
                and (a == Constant(1)
                     or (b == Constant(0) and c == Constant(0)
                     and h == Constant(0) and i == Constant(0))))

    def apply_const_on_dur(a: Feature, b: Feature):
        return a == Constant(0) or b == Constant(0)

    def apply_const_on_dur_dbyte(a: Feature, b: Feature, c: Feature):
        return a > Constant(0) or b == Constant(1) or c == Constant(0)

    # if proto != tcp, then all tcp fields should be 0
    g1 = apply_const_on_tcp_fields(Feature(1), Feature(18), Feature(21),
                                   Feature(12), Feature(8), Feature(0),
                                   Feature(9), Feature(19), Feature(20))

    # if dur=0, then dbytes=0
    g2 = apply_const_on_dur(Feature(0), Feature(12))

    # if dur > 0 and stat=INT, then dbytes = 0
    g3 = apply_const_on_dur_dbyte(Feature(0), Feature(9), Feature(12))

    return [g1, g2, g3]


def get_iot_constraints() -> List[BaseRelationConstraint]:
    # TODO: implement
    # there must be at least two constraints to compute loss
    def apply_const_on_s0state(a: Feature, b: Feature, c: Feature):
        return (a != Constant(1)) or (b == c == Constant(0))

    def apply_const_on_orig_ip_bytes(a: Feature, b: Feature, c: Feature, d: Feature):
        return ((a != Constant(1))
                or b == Constant(0) or c >= Constant(20)
                or g5 or g6 or d != 1)

    def apply_const_on_resp_pkt(a: Feature, b: Feature, c: Feature):
        return ((a != Constant(1))
                or g5 or (b == Constant(1) and c == Constant(1))
                or g7 or g8 or c == Constant(1))

    # orig_pkts <= orig_ip_bytes
    g1 = Feature(14) <= Feature(15)

    # resp_pkts <= resp_ip_bytes
    g2 = Feature(16) <= Feature(17)

    # when the connection state is S0, there is no response packet and bytes
    g3 = apply_const_on_s0state(Feature(6), Feature(16), Feature(17))

    # ori_packets > 0 and  ori_bytes > 20
    g4 = apply_const_on_orig_ip_bytes(Feature(1), Feature(14), Feature(15), Feature(8))

    # orig_pkts >= resp_pkts
    g5 = Feature(14) >= Feature(16)

    # orig_ip_bytes < resp_ip_bytes
    g6 = Feature(15) < Feature(17)

    # orig_pkts < resp_pkts
    g7 = Feature(14) < Feature(16)

    # orig_ip_bytes >= resp_ip_bytes
    g8 = Feature(15) >= Feature(17)

    # constraints when proto is not udp
    g9 = apply_const_on_resp_pkt(Feature(0), Feature(10), Feature(7))

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9]

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


def cpgd_apply_and_predict(
        keras_nn: Sequential, x: np.ndarray, y: np.ndarray,
        enable_constr: bool, feat_file: str, **config
):
    args_ = {**config['args'], 'enable_constraints': enable_constr}
    constraints = get_constraints_from_file(
        *init_constraints(feat_file))
    scaler = get_scaler(feat_file)
    model = TensorflowClassifier(keras_nn)  # wrap in their interface
    pipe = Pipeline(steps=[("preprocessing", scaler), ("model", model)])
    x_adv = CPGD(pipe, constraints, **args_).generate(x, y)
    x_adv = x_adv.reshape(-1, x_adv.shape[-1])  # remove extra axis
    y_adv = model.predict(x_adv)
    return x_adv, y_adv
