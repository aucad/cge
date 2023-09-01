import numpy as np
import pandas as pd

from keras.models import Sequential
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
        keras_nn: Sequential, x: np.ndarray, y: np.ndarray,
        feat_file: str, **config
):
    args_ = {**config['args'], 'enable_constraints': True}
    constraints = get_constraints_from_file(feat_file, [])
    scaler = get_scaler(feat_file)
    model = TensorflowClassifier(keras_nn)  # wrap in their interface
    pipe = Pipeline(steps=[("preprocessing", scaler), ("model", model)])
    x_adv = CPGD(pipe, constraints, **args_).generate(x, y)
    x_adv = x_adv.reshape(-1, x_adv.shape[-1])  # remove the extra axis
    y_adv = model.predict(x_adv)
    return x_adv, y_adv
