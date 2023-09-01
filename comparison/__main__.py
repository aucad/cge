import numpy as np
import pandas as pd

from keras.models import load_model, Sequential
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from . import CPGD, TensorflowClassifier, get_constraints_from_file



def get_scaler(path: str):
    df = pd.read_csv(path, low_memory=False)
    min_c = df["min"].to_numpy()
    max_c = df["max"].to_numpy()
    _scaler = MinMaxScaler(feature_range=(0, 1))
    _scaler.fit([min_c, max_c])
    return _scaler


def cpgd_apply_predict(
        internal_model: Sequential,
        x: np.ndarray, y: np.ndarray,
        enable_constr: bool, feat_file: str, **config
):
    args_ = {**config['args'], 'enable_constraints': enable_constr}
    constraints = get_constraints_from_file(feat_file, [])
    scaler = get_scaler(feat_file)
    model_ = TensorflowClassifier(internal_model)
    pipe = Pipeline(steps=[("preprocessing", scaler), ("model", model_)])
    attack = CPGD(pipe, constraints, **args_)
    x_adv = attack.generate(x, y)
    y_adv = y.copy()
    return x_adv, y_adv


if __name__ == '__main__':
    from .constr import get_url_relation_constraints

    RES_DIR = "./comparison/resources/"
    X_CLEAN = f"{RES_DIR}baseline_X_test_candidates.npy"
    Y_CLEAN = f"{RES_DIR}baseline_y_test_candidates.npy"
    FEATURES = f"{RES_DIR}features.csv"
    MODEL = f"{RES_DIR}baseline_nn.model"
    x_clean = np.load(X_CLEAN)[:32]
    y_clean = np.load(Y_CLEAN)[:32]
    constr = get_constraints_from_file(
        FEATURES, get_url_relation_constraints())
    model = TensorflowClassifier(load_model(MODEL))
    args = {
        "norm": 2,
        "eps": 0.2,
        "eps_step": 0.1,
        "save_history": None,
        "seed": None,
        "n_jobs": -1,
        "verbose": 1,
        "enable_constraints": True}

    model_pipeline = Pipeline(steps=[
        ("preprocessing", get_scaler(FEATURES)),
        ("model", model)])

    res = CPGD(model_pipeline, constr, **args).generate(x_clean, y_clean)
    print(f'generated {len(res)} examples')
