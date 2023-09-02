# noinspection PyPackageRequirements
from art.estimators.classification import TensorFlowV2Classifier
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam

from exp import TargetModel
from exp.utility import clear_console_lines


class DeepNeuralNetwork(TargetModel):

    def __init__(self, conf):
        super().__init__(conf)
        self.name = "Neural Network"

    def train(self):
        n_layers = self.conf['layers']
        fit_args = self.conf['model_fit']
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
        self.classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=self.n_classes,
            loss_object=model.loss,
            input_shape=self.train_x.shape[1:],
            clip_values=(0, 1))
        self.model = self.classifier.model
        predictions = self.predict(self.test_x, self.test_y)
        self.score.calculate(self.test_y, predictions)
        clear_console_lines()
        return self

    def predict(self, x, y):
        tmp = self.model.predict(x)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)
