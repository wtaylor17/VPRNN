import numpy as np
np.random.seed(314159)  # for reproducibility with perm
try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import set_random_seed as set_seed
set_seed(21)

from vprnn.models import load_vprnn
from vprnn.mnist_data import load_mnist_stash

from keras.utils import to_categorical

import sys


T = 28*28
# load MNIST
(x_train, y_train), (x_test, y_test) = load_mnist_stash()
# scale down to [-1,1], reshape, permute
x_test = x_test.reshape(x_test.shape[0], -1, 1)
P = np.random.permutation(T)
print(P.tolist())
x_test = x_test[:, P, :]
x_test = x_test.astype('float32')
x_test /= 255
x_test -= .5
x_test *= 2
y_test = to_categorical(y_test, 10)


model_name = sys.argv[1]

model = load_vprnn(model_name, do_compile=False)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.summary()
pred = model.predict(x_test)

n_test = x_test.shape[0]
n_correct = 0
for y, p in zip(pred, y_test):
    if np.argmax(y) == np.argmax(p):
        n_correct += 1
print(f'Final Test accuracy = {100 * n_correct / n_test}%.')
