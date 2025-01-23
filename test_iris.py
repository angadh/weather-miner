import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import ucimlrepo

class TestIris(unittest.TestCase):
    def test_iris(self):

        iris = ucimlrepo.fetch_ucirepo(id=53)
        X = iris.data.features.values

        targets = iris.data.targets
        targets.loc[:, 'label'] = iris.data.targets.rank(method='dense').astype(int) - 1
        y = keras.utils.to_categorical(targets[['label']].values, num_classes = 3)

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(4,), name='input_layer'))
        model.add(keras.layers.Dense(10, activation='relu', name='hidden_layer_1'))
        model.add(keras.layers.Dense(10, activation='relu', name='hidden_layer_2'))
        model.add(keras.layers.Dense(3, activation='softmax', name='output_layer'))

        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        model.fit(X, y, epochs=50, verbose=1)
        model_accuracy = model.evaluate(X, y.astype(np.float32))[1]

        self.assertGreater(model_accuracy, 0.9)

if __name__ == "__main__":
    unittest.main()
