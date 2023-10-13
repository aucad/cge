"""
Implementation of constrained adversarial attacks using CPGD.
This implementation uses a different constraint evaluation strategy,
and the constraint checking is built into the attack. See:

Paper: https://arxiv.org/abs/2112.01156
Source code: https://github.com/serval-uni-lu/constrained-attacks
Experiments: https://github.com/serval-uni-lu/moeva2-ijcai22-replication

The comparison implementation is in comparison/ under its own license.
Minor modifications were applied to make it compatible with more
recent versions of Python and adversarial-robustness-toolkit than the
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


def get_perf1_constraints() -> List[BaseRelationConstraint]:
    g1 = Feature(9) != Constant(1) or Feature(12) == Constant(0)
    g2 = Feature(6) != Constant(1) or Feature(22) <= Constant(512)
    g3 = Feature(12) != Constant(0) or Feature(15) == Constant(0)
    return [g1, g2, g3]


def get_perf2_constraints() -> List[BaseRelationConstraint]:
    g4 = (Feature(4) != Constant(1) or Feature(3) != Constant(1) or
          Feature(27) >= Constant(1))
    g5 = Feature(8) != Constant(1) or Feature(28) >= Constant(1)
    g6 = Feature(23) >= Constant(10) or Feature(16) >= Constant(17)
    return get_perf1_constraints() + [g4, g5, g6]


def get_perf3_constraints() -> List[BaseRelationConstraint]:
    g7 = Feature(9) != Constant(1) or Feature(22) >= Feature(23)
    g8 = (Feature(24) != Constant(1) or Feature(8) == Constant(1) or
          Feature(10) == Constant(1))
    g9 = ((Feature(18) == Constant(0) and Feature(21) == Constant(0))
          or Feature(23) >= Constant(0))
    g10 = Feature(8) != Constant(1) or Feature(22) >= Constant(1)
    g11 = (Feature(10) != Constant(1) or
           (Feature(22) >= Constant(0) and
            Feature(23) >= Constant(0)))
    g12 = Feature(12) >= Constant(0) and Feature(15) >= Constant(0)
    return get_perf2_constraints() + [g7, g8, g9, g10, g11, g12]


def get_perf4_constraints() -> List[BaseRelationConstraint]:
    g1 = Constant(0) <= Feature(18) <= Constant(255)
    g2 = Constant(0) <= Feature(21) <= Constant(255)
    g3 = Constant(0) <= Feature(14) <= Constant(3993)
    return [g1, g2, g3]


def get_perf5_constraints() -> List[BaseRelationConstraint]:
    g4 = Constant(0) <= Feature(15) <= Constant(2627)
    g5 = Constant(0) <= Feature(11) <= Constant(7085342)
    g6 = Constant(0) <= Feature(12) <= Constant(10508068)
    return get_perf4_constraints() + [g4, g5, g6]


def get_perf6_constraints() -> List[BaseRelationConstraint]:
    g7 = Constant(0) <= Feature(0) <= Constant(60)
    g8 = Constant(1) <= Feature(25) <= Constant(63)
    g9 = Constant(1) <= Feature(26) <= Constant(50)
    g10 = Constant(1) <= Feature(27) <= Constant(50)
    g11 = Constant(1) <= Feature(28) <= Constant(46)
    g12 = Constant(1) <= Feature(29) <= Constant(63)
    return get_perf5_constraints() + [g7, g8, g9, g10, g11, g12]


def get_unsw_constraints() -> List[BaseRelationConstraint]:
    g1 = Feature(1) + Feature(2) + Feature(3) == Constant(1)

    g2 = (Feature(4) + Feature(5) + Feature(6) + Feature(7) ==
          Constant(1))

    g3 = Feature(8) + Feature(9) + Feature(10) == Constant(1)

    g4 = ((Feature(1)) != Constant(1) or
          (Feature(18) == Constant(255) or
           Feature(21) == Constant(255)) or
          (Feature(12) == Feature(21) == Constant(0)) or
          Feature(8) == Constant(1))

    g5 = ((Feature(1)) != Constant(1) or
          Feature(8) == Constant(1) or
          Feature(0) == Constant(0) or
          Feature(9) != Constant(1) or
          Feature(12) == Constant(0))

    g6 = (Feature(1) == Constant(1) or
          (Feature(18) == Feature(19) == Feature(20) ==
           Feature(21) == 0))

    return [g1, g2, g3, g4, g5, g6]


def get_iot_constraints() -> List[BaseRelationConstraint]:
    g1 = Feature(14) <= Feature(15)

    g2 = Feature(16) <= Feature(17)

    g3 = ((Feature(6) != Constant(1)) or
          (Feature(16) == Feature(17) == Constant(0)))

    g4 = (Feature(1) != Constant(1) or
          Feature(14) == Constant(0) or
          Feature(15) >= Constant(20))

    g5 = (Feature(1) != Constant(1) or
          Feature(16) == Constant(0) or
          Feature(17) >= Constant(20))

    g6 = (Feature(1) != Constant(1) or
          Feature(14) >= Feature(16) or
          Feature(15) < Feature(17) or
          Feature(8) != Constant(1))

    g7 = (Feature(0) != Constant(1) or
          Feature(14) >= Feature(16) or
          (Feature(11) == Constant(1) and
           Feature(7) == Constant(1)))

    g8 = (Feature(0) != Constant(1) or
          Feature(14) < Feature(16) or
          Feature(15) >= Feature(17) or
          Feature(7) == Constant(1))

    return [g1, g2, g3, g4, g5, g6, g7, g8]


def get_lcld_constraints() -> List[BaseRelationConstraint]:
    """from <https://tinyurl.com/cfvpjhwu>"""
    tol = Constant(1e-3)

    ir_1200 = Feature(2) / Constant(1200)
    ir_1200_p1 = Constant(1) + ir_1200
    g1 = (((Feature(3) - (
            (Feature(0) * ir_1200 * (ir_1200_p1 ** Feature(1))) /
            ((ir_1200_p1 ** Feature(1)) - Constant(1)))) **
           Constant(2) ** Constant(0.5) - Constant(0.099999))
          <= Constant(20))

    # open_acc <= total_acc
    g2 = Feature(10) <= Feature(14)

    # pub_rec_bankruptcies <= pub_rec
    g3 = Feature(16) <= Feature(11)

    # term = 36 or term = 60
    term_val = Feature(1) ** Constant(2) ** Constant(0.5)
    g4 = (term_val <= Constant(36) + tol or term_val <= Constant(
        60) + tol)

    # # ratio_loan_amnt_annual_inc
    g5 = (((Feature(20) - Feature(0) / Feature(6)) ** Constant(2))
          ** Constant(0.5)) <= tol

    # # ratio_open_acc_total_acc
    # g6 = np.absolute(Feature(21) - Feature(10) / Feature(14))
    g6 = (((Feature(21) - Feature(10) / Feature(14)) ** Constant(2))
          ** Constant(0.5)) <= tol

    # diff_issue_d_earliest_cr_line
    def date_ft_to_month(x):
        return Feature(x) / Constant(100) * Constant(12) \
               + (Feature(x) % Constant(100))

    g7 = Feature(22) - (
            date_ft_to_month(7) - date_ft_to_month(9)) <= tol

    # ratio_pub_rec_diff_issue_d_earliest_cr_line
    g8 = (((Feature(23) - Feature(11) / Feature(22)) ** Constant(2))
          ** Constant(0.5)) <= tol

    # ratio_pub_rec_bankruptcies_pub_rec
    g9 = (((Feature(24) - Feature(16) / Feature(22)) ** Constant(2))
          ** Constant(0.5)) <= tol

    g10 = ((Feature(25) - Feature(16) / (Feature(11) + Constant(1e-5)))
           ** Constant(2) ** Constant(0.5)) <= tol

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


def get_url_constraints() -> List[BaseRelationConstraint]:
    """from <https://tinyurl.com/zdypm9a8>"""

    def apply_if_a_supp_zero_than_b_supp_zero(a: Feature, b: Feature):
        return (Constant(0) <= a) or (Constant(0) < b)

    g1 = Feature(1) <= Feature(0)

    intermediate_sum = Constant(0)
    for i in range(3, 18):
        intermediate_sum = intermediate_sum + Feature(i)
    intermediate_sum = intermediate_sum + (Constant(3) * Feature(19))

    g2 = intermediate_sum <= Feature(0)

    # g3: if x[:, 21] > 0 then x[:,3] > 0
    g3 = apply_if_a_supp_zero_than_b_supp_zero(Feature(21), Feature(3))

    # g4: if x[:, 23] > 0 then x[:,13] > 0
    g4 = apply_if_a_supp_zero_than_b_supp_zero(Feature(23), Feature(13))

    intermediate_sum = \
        (Constant(3) * Feature(20) + Constant(4) * Feature(21)
         + Constant(2) * Feature(23))
    g5 = intermediate_sum <= Feature(0)

    # g6: if x[:, 19] > 0 then x[:,25] > 0
    g6 = apply_if_a_supp_zero_than_b_supp_zero(Feature(19), Feature(25))

    # g8: if x[:, 2] > 0 then x[:,25] > 0
    g8 = apply_if_a_supp_zero_than_b_supp_zero(Feature(2), Feature(25))

    # g10: if x[:, 28] > 0 then x[:,25] > 0
    g10 = apply_if_a_supp_zero_than_b_supp_zero(Feature(28),
                                                Feature(25))

    # g11: if x[:, 31] > 0 then x[:,26] > 0
    g11 = apply_if_a_supp_zero_than_b_supp_zero(Feature(31),
                                                Feature(26))

    # x[:,38] <= x[:,37]
    g12 = Feature(38) <= Feature(37)
    g13 = (Constant(3) * Feature(20)) <= (Feature(0) + Constant(1))
    g14 = (Constant(4) * Feature(21)) <= (Feature(0) + Constant(1))
    g15 = (Constant(4) * Feature(2)) <= (Feature(0) + Constant(1))
    g16 = (Constant(2) * Feature(23)) <= (Feature(0) + Constant(1))

    return [g1, g2, g3, g4, g5, g6, g8, g10, g11, g12,
            g13, g14, g15, g16]


def init_constraints(feat_file, key=None):
    c_set, match_pattern = None, key or feat_file
    if 'perf1' in match_pattern:
        c_set = get_perf1_constraints()
    elif 'perf2' in match_pattern:
        c_set = get_perf2_constraints()
    elif 'perf3' in match_pattern:
        c_set = get_perf3_constraints()
    elif 'perf4' in match_pattern:
        c_set = get_perf4_constraints()
    elif 'perf5' in match_pattern:
        c_set = get_perf5_constraints()
    elif 'perf6' in match_pattern:
        c_set = get_perf6_constraints()
    elif 'unsw' in match_pattern:
        c_set = get_unsw_constraints()
    elif 'iot' in match_pattern:
        c_set = get_iot_constraints()
    elif 'lcld' in match_pattern:
        c_set = get_lcld_constraints()
    elif 'url' in match_pattern:
        c_set = get_url_constraints()
    if not c_set or len(c_set) < 2:
        g1 = Feature(0) <= Feature(0)
        g2 = Feature(1) <= Feature(1)
        c_set = [g1, g2]
        print('Using default minimal constraints for CPGD')
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
        feat_file: str, args: dict, **kwargs
):
    const_key = kwargs['id'] if 'id' in kwargs else None
    constraints = get_constraints_from_file(
        *init_constraints(feat_file, const_key))
    scaler = get_scaler(feat_file)
    model = TensorflowClassifier(keras_nn)  # wrap in their interface
    pipe = Pipeline(steps=[("preprocessing", scaler), ("model", model)])
    x_adv = CPGD(pipe, constraints, **args).generate(x, y)
    x_adv = x_adv.reshape(-1, x_adv.shape[-1])  # remove extra axis
    y_adv = model.predict(x_adv)
    return x_adv, y_adv
