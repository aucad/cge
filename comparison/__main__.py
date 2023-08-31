import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from keras.models import load_model

from . import CPGD, TensorflowClassifier, get_constraints_from_file
from .constr import get_url_relation_constraints


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
