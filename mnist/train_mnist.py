import tensorflow as tf
from tensorflow import keras

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

def train_mnist():

    x_train, y_train, x_test, y_test = get_mnist_data()
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),

        keras.layers.Dense(32, activation='relu', name='hidden_layer_1'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(16, activation='relu', name='embedding_layer'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(10, activation='softmax', name='output_layer')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train_cat, epochs=100, verbose=1) 

    model.evaluate(x_test, y_test_cat)
    return model

if __name__ == '__main__':
    model = train_mnist()

    embedding_layer = model.get_layer('embedding_layer')
    embedding_layer.save('embedding_layer.keras')
