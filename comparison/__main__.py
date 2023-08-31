import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from keras.models import load_model
from typing import List

from . import CPGD, TensorflowClassifier
from . import get_constraints_from_file, Constant, Feature, \
    BaseRelationConstraint


def get_url_relation_constraints() -> List[BaseRelationConstraint]:
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

    intermediate_sum = (
            Constant(3) * Feature(20) + Constant(4) * Feature(21) +
            Constant(2) * Feature(23))
    g5 = intermediate_sum <= Feature(0)

    # g6: if x[:, 19] > 0 then x[:,25] > 0
    g6 = apply_if_a_supp_zero_than_b_supp_zero(Feature(19), Feature(25))

    # g8: if x[:, 2] > 0 then x[:,25] > 0
    g8 = apply_if_a_supp_zero_than_b_supp_zero(Feature(2), Feature(25))

    # g10: if x[:, 28] > 0 then x[:,25] > 0
    g10 = apply_if_a_supp_zero_than_b_supp_zero(Feature(28), Feature(25))

    # g11: if x[:, 31] > 0 then x[:,26] > 0
    g11 = apply_if_a_supp_zero_than_b_supp_zero(Feature(31), Feature(26))

    # x[:,38] <= x[:,37]
    g12 = Feature(38) <= Feature(37)
    g13 = (Constant(3) * Feature(20)) <= (Feature(0) + Constant(1))
    g14 = (Constant(4) * Feature(21)) <= (Feature(0) + Constant(1))
    g15 = (Constant(4) * Feature(2)) <= (Feature(0) + Constant(1))
    g16 = (Constant(2) * Feature(23)) <= (Feature(0) + Constant(1))

    return [g1, g2, g3, g4, g5, g6, g8, g10, g11, g12, g13, g14, g15, g16]


def run_cpgd_attack(X, y, constraints, enable_constraints: bool):
    model = TensorflowClassifier(load_model(MODEL))
    preprocessing_pipeline = joblib.load(JOBLIB)
    model_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing_pipeline),
        ("model", model)])
    attack = CPGD(
        model_pipeline, constraints,
        norm=2,
        eps=0.2,
        eps_step=0.1,
        save_history=None,
        seed=None,
        n_jobs=-1,
        verbose=1,
        enable_constraints=enable_constraints,
    )
    x_adv = attack.generate(X, y)
    return x_adv


if __name__ == '__main__':
    RES_DIR = "./comparison/resources/"
    X_CLEAN = f"{RES_DIR}baseline_X_test_candidates.npy"
    Y_CLEAN = f"{RES_DIR}baseline_y_test_candidates.npy"
    MODEL = f"{RES_DIR}baseline_nn.model"
    JOBLIB = f"{RES_DIR}baseline_scaler.joblib"
    FEATURES = f"{RES_DIR}features.csv"

    x_clean = np.load(X_CLEAN)[:32]
    y_clean = np.load(Y_CLEAN)[:32]
    constr = get_constraints_from_file(
        FEATURES, get_url_relation_constraints())
    run_cpgd_attack(x_clean, y_clean, constr, True)
    run_cpgd_attack(x_clean, y_clean, constr, False)
