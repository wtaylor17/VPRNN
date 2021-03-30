from vprnn.har_data import load_har_stash
from vprnn.models import load_vprnn

import sys
import numpy as np


np.random.seed(12345)

model = load_vprnn(sys.argv[1])
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['acc'])

if len(sys.argv) == 2:
    (x_train, y_train), (x_test, y_test) = load_har_stash()
else:
    (x_train, y_train), (x_test, y_test) = load_har_stash(sys.argv[2])
mu = np.mean(x_train, axis=(0, 1)).reshape((1, 1, -1))
std = np.std(x_train, axis=(0, 1)).reshape((1, 1, -1))
x_train = (x_train - mu) / std
x_test = (x_test - mu) / std
print(model.evaluate(x_test, y_test))
