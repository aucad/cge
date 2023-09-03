# noinspection PyPackageRequirements
from art.estimators.classification import TensorFlowV2Classifier
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam

from exp import BaseModel


class DeepNeuralNetwork(BaseModel):

    def __init__(self, conf):
        super().__init__(conf)
        self.name = "Neural Network"

    def predict(self, x, y):
        tmp = self.model.predict(x, verbose=0)
        return tmp.argmax(axis=1 if len(tmp.shape) == 2 else 0)

    def train_steps(self):
        n_layers, fit = self.conf['layers'], self.conf['model_fit']
        self.model = Sequential(
            [Dense(v, activation='relu') for v in n_layers] +
            [Dense(self.n_classes, activation='softmax')])
        self.model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()])
        self.model.fit(
            self.train_x, self.train_y, **fit,
            callbacks=[EarlyStopping(monitor='loss', patience=5)])
        self.classifier = TensorFlowV2Classifier(
            model=self.model,
            nb_classes=self.n_classes,
            loss_object=self.model.loss,
            input_shape=self.train_x.shape[1:],
            clip_values=(0, 1))
