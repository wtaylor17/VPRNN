import keras
import numpy as np

import os


SCRIPT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, 'mnist-stash')


def stash_mnist(directory=DEFAULT_DIRECTORY):
    os.makedirs(directory, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    def joiner(s):
        return os.path.join(directory, s)
    np.save(joiner('x_train.npy'), x_train)
    np.save(joiner('y_train.npy'), y_train)
    np.save(joiner('x_test.npy'), x_test)
    np.save(joiner('y_test.npy'), y_test)


def load_mnist_stash(directory=DEFAULT_DIRECTORY):
    def joiner(s):
        return os.path.join(directory, s)
    return (np.load(joiner('x_train.npy')), np.load(joiner('y_train.npy'))),\
           (np.load(joiner('x_test.npy')), np.load(joiner('y_test.npy')))
