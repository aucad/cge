from exp import DeepNeuralNetwork, XGBoost


class ClsPicker:
    XGB = 'xgb'
    DNN = 'dnn'

    @staticmethod
    def list_cls():
        return sorted([ClsPicker.XGB, ClsPicker.DNN])

    @staticmethod
    def load(name):
        if name == ClsPicker.XGB:
            return XGBoost
        if name == ClsPicker.DNN:
            return DeepNeuralNetwork
