import tensorflow as tf
from art.estimators.classification import KerasClassifier
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from exp import ModelTraining

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DeepNeuralNetwork(ModelTraining):

    def __init__(self, conf):
        super().__init__(conf)
        self.name = "Neural Network"

    def train(self):
        n_layers = self.conf['layers']
        fit_args = self.conf['model_fit']
        model_args = self.conf['params']

        layers = [Dense(v, activation='relu') for v in n_layers] + \
                 [Dense(self.n_classes, activation='softmax')]
        model = Sequential(layers)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()])
        model.fit(
            self.train_x, self.train_y, **fit_args,
            callbacks=[EarlyStopping(monitor='loss', patience=5)])
        self.classifier = KerasClassifier(
            model=model, **model_args, clip_values=(0, 1))
        self.model = self.classifier.model
        predictions = self.predict(self.test_x, self.test_y)
        self.score.calculate(self.test_y, predictions)
        return self

    def predict(self, x, y):
        tmp = self.model.predict(x)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)
