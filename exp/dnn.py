import tensorflow as tf

from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from art.estimators.classification import KerasClassifier

from exp import ModelTraining

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DeepNeuralNetwork(ModelTraining):

    def __init__(self, conf):
        super().__init__(conf)
        self.name = "Neural Network"

    def train(self):
        layers = self.conf['layers']
        fit_args = self.conf['model_fit']
        layers = [Dense(v, activation='relu') for v in layers] + \
                 [Dense(self.n_classes, activation='softmax')]
        model = tf.keras.models.Sequential(layers)
        model.compile(
            optimizer=SGD(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()])
        model.fit(
            self.train_x, self.train_y, **fit_args,
            callbacks=[EarlyStopping(monitor='loss', patience=5)])
        self.classifier = \
            KerasClassifier(model=model, clip_values=(0, 1))
        predictions = self.predict(self.test_x, self.test_y)
        self.score.calculate(self.test_y, predictions)
        return self

    def predict(self, x, y):
        tmp = self.model.predict(x, y)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)
