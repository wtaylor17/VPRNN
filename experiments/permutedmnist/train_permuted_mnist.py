# reproduce same as smnist
import numpy as np

np.random.seed(314159)
try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import set_random_seed as set_seed
set_seed(21)


from keras.layers import Input, RNN, Dense, Dropout, LSTMCell
from keras.utils import to_categorical
from keras.models import Model
from keras import optimizers
from datetime import datetime
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from keras_layer_normalization import LayerNormalization

import os
import json
import sys
from argparse import ArgumentParser
from math import log2, ceil

sys.path.append('../../')
from vprnn.layers import VanillaCell
from vprnn.models import save_vprnn
from vprnn.mnist_data import load_mnist_stash

parser = ArgumentParser()
parser.add_argument('--stamp', '-s', type=str,
                    default=str(datetime.now()).replace(' ', '-').replace(':', '.'))
parser.add_argument('--activation', '-a', type=str, default='relu')
parser.add_argument('--dim', '-d', type=int, default=128)
parser.add_argument('--rotations', '-r', type=int, default=None)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--noclip', '-c', action='store_true')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
parser.add_argument('--tensorboard', '-t', action='store_true')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--rnn_type', type=str, default='vanilla')
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--optimizer', '-o', type=str, default='adam')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--nodiag',
                    action='store_true')
parser.add_argument('--use_ln', action='store_true')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--input_dropout', type=float, default=0.0)
parser.add_argument('--verbose', type=int, default=2)
parser.add_argument('--test_mode', action='store_true')

args = parser.parse_args()

testing = args.test_mode
T = 28 * 28
stamp = args.stamp
recurrent_activation = args.activation
H = args.dim
K = args.rotations or ceil(log2(H))
epochs = args.epochs
recurrent_clip = -1 if args.noclip else (2 ** (1 / T))
initial_lr = args.learning_rate
tensorboard = args.tensorboard
n_layers = args.n_layers
model_type = args.rnn_type
batch_size = args.batch_size
dropout = args.dropout
use_ln = args.use_ln

# load MNIST
(x_train, y_train), (x_test, y_test) = load_mnist_stash()
# scale down to [-1,1], reshape, permute
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
perm = np.random.permutation(T)
x_train = x_train[:, perm, :]
x_test = x_test[:, perm, :]
x_train /= 255
x_test /= 255
x_train -= .5
x_test -= .5
x_train *= 2
x_test *= 2
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

if testing:
    x_val, y_val = x_test, y_test
else:
    num_train = 48000
    x_val, y_val = x_train[num_train:], y_train[num_train:]
    x_train, y_train = x_train[:num_train], y_train[:num_train]

assert args.optimizer in ['adam', 'rmsprop'], 'only adam + rmsprop supported'
opt_cls = optimizers.Adam if args.optimizer == 'adam' else optimizers.RMSprop
optimizer = opt_cls(lr=initial_lr)

input_tensor = Input((T, 1))
rnn_out = input_tensor
if 0 < args.input_dropout < 1:
    rnn_out = Dropout(args.input_dropout,
                      noise_shape=(None, 1, 1))(rnn_out)
rnn_kwargs = dict(n_rotations=K,
                  activation=recurrent_activation,
                  recurrent_clip=recurrent_clip,
                  use_diag=not args.nodiag)

assert model_type in ['vanilla', 'lstm'], "only vanilla and lstm cells supported"
if model_type == 'vanilla':
    rnn_cls = VanillaCell
else:
    rnn_cls = LSTMCell
    rnn_kwargs = {}
print('using rnn class', rnn_cls)
for _ in range(n_layers - 1):
    rnn_out = RNN(rnn_cls(H, **rnn_kwargs),
                  return_sequences=True)(rnn_out)
    if use_ln:
        rnn_out = LayerNormalization()(rnn_out)
    if dropout > 0:
        rnn_out = Dropout(dropout, noise_shape=(None, 1, H))(rnn_out)
rnn_out = RNN(rnn_cls(H, **rnn_kwargs))(rnn_out)
dense = Dense(10, activation='softmax')

model = Model(input_tensor, dense(rnn_out))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
model.summary()

callbacks = []
if tensorboard:
    callbacks.append(TensorBoard(log_dir=f'tblogs/{stamp}'))

if not testing:
    callbacks.append(EarlyStopping(monitor='val_acc',
                                   patience=args.patience,
                                   mode='max',
                                   restore_best_weights=True))
else:
    def scheduler(epoch, lr):
        hundreds = (epoch + 1) // 100
        return initial_lr / (2 ** hundreds)

    callbacks.append(LearningRateScheduler(scheduler))

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    validation_data=[x_val, y_val],
                    callbacks=callbacks,
                    batch_size=batch_size,
                    verbose=args.verbose).history

if not os.path.exists(f'models/{stamp}'):
    os.makedirs(f'models/{stamp}')

# if "testing" is true, this is a test evaluation (otherwise validation)
evaluation = model.evaluate(x_val, y_val)
print('EVALUATION:', evaluation)

if model_type == 'vanilla' and testing:
    save_vprnn(model, f'models/{stamp}')

with open(f'models/{stamp}/run_config.json', 'w') as json_fp:
    json.dump({
        'K': K,
        'T': T,
        'H': H,
        'epochs': epochs,
        'recurrent_activation': recurrent_activation,
        'initial_lr': initial_lr,
        'optimizer': type(optimizer).__name__,
        'tensorboard': tensorboard,
        'recurrent_clip': recurrent_clip,
        'ln': use_ln,
        'rnn_kwargs': rnn_kwargs,
        'test_evaluation': list(evaluation)
    }, json_fp, indent=4)
