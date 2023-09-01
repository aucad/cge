import numpy as np
import pandas as pd

from keras.models import load_model, Sequential
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from . import CPGD, TensorflowClassifier, get_constraints_from_file as cff


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
    constraints = cff(feat_file, [])
    scaler = get_scaler(feat_file)
    model = TensorflowClassifier(keras_nn)  # wrap in their interface
    pipe = Pipeline(steps=[("preprocessing", scaler), ("model", model)])
    x_adv = CPGD(pipe, constraints, **args_).generate(x, y)
    x_adv = x_adv.reshape(-1, x_adv.shape[-1])  # remove the extra axis
    y_adv = model.predict(x_adv)
    return x_adv, y_adv


def url_pretrained_example():
    from .constr import get_url_relation_constraints

    RES_DIR = "./comparison/example/"
    X_CLEAN = f"{RES_DIR}baseline_X_test_candidates.npy"
    Y_CLEAN = f"{RES_DIR}baseline_y_test_candidates.npy"
    FEATURES = f"{RES_DIR}features.csv"
    MODEL = f"{RES_DIR}baseline_nn.model"

    x_clean, y_clean = np.load(X_CLEAN)[:32], np.load(Y_CLEAN)[:32]
    constr = cff(FEATURES, get_url_relation_constraints())
    model_ = TensorflowClassifier(load_model(MODEL))
    model_pipeline = Pipeline(steps=[
        ("preprocessing", get_scaler(FEATURES)), ("model", model_)])

    adv_x = CPGD(
        model_pipeline, constr,
        norm=2,
        eps=0.2,
        eps_step=0.1,
        save_history=None,
        seed=None,
        n_jobs=-1,
        verbose=1,
        enable_constraints=True,
    ).generate(x_clean, y_clean)

    y_pred = model_.predict(adv_x.reshape(-1, adv_x.shape[-1]))
    evades = np.sum(y_pred != y_clean)

    print(f'generated {len(adv_x)} examples, evades: ', evades)


if __name__ == '__main__':
    url_pretrained_example()
