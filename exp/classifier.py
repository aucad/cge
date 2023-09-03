from exp import DeepNeuralNetwork, XGBoost
from exp.utility import upper_attrs


class ClsPicker:
    XGB = 'xgb'
    DNN = 'dnn'

    @staticmethod
    def list_cls():
        return upper_attrs(ClsPicker)

    @staticmethod
    def load(name):
        if name == ClsPicker.XGB:
            return XGBoost
        if name == ClsPicker.DNN:
            return DeepNeuralNetwork
