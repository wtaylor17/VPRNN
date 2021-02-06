from vprnn.layers import VanillaCell
from keras.layers import RNN, Dense, Input, Dropout, LSTMCell
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras_layer_normalization import LayerNormalization
import argparse
import numpy as np
from vprnn.models import save_vprnn

from math import ceil, log2

from vprnn.har_data import load_har_stash


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_rotations', type=int, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--input_dropout', type=float, default=0.0)
parser.add_argument('--clf_dropout', type=float, default=0.0)
parser.add_argument('--use_ln', action='store_true')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--model_name', type=str, default='har_model')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--optimizer', type=str, default='rmsprop')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cell', type=str, default='vanilla')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--no_validation', action='store_true')
parser.add_argument('--decay', type=float, default=0)

args = parser.parse_args()


def scheduler(epoch, lr):
    return args.lr / (10 ** (epoch // 100))


callbacks: list = [LearningRateScheduler(scheduler)] if args.no_validation else []

np.random.seed(args.seed)
(x_train, y_train), (x_test, y_test) = load_har_stash()
mu = np.mean(x_train, axis=(0, 1)).reshape((1, 1, -1))
std = np.std(x_train, axis=(0, 1)).reshape((1, 1, -1))
x_train = (x_train - mu) / std
x_test = (x_test - mu) / std
perm = np.random.permutation(x_train.shape[0])
x_train, y_train = x_train[perm], y_train[perm]
n_val = int(0.2 * x_train.shape[0])
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

x_val, y_val = x_test, y_test
if not args.no_validation:
    x_train, x_val = x_train[:-n_val], x_train[-n_val:]
    y_train, y_val = y_train[:-n_val], y_train[-n_val:]
    print('Positive ratios:')
    print('train:', y_train.reshape((-1,)).tolist().count(1) / y_train.shape[0])
    print('val:', y_val.reshape((-1,)).tolist().count(1) / y_val.shape[0])
    print('test:', y_test.reshape((-1,)).tolist().count(1) / y_test.shape[0])
    callbacks.append(EarlyStopping(monitor='val_acc',
                                   mode='max',
                                   restore_best_weights=True,
                                   patience=args.patience))

inp = Input((None, x_train.shape[-1]))
rnn_out = inp

if 0 < args.input_dropout < 1:
    rnn_out = Dropout(args.input_dropout,
                      noise_shape=(None, 1, x_train.shape[-1]))(rnn_out)


def cell():
    if args.cell == 'vanilla':
        cell_ = VanillaCell(args.hidden_dim,
                            use_diag=False,
                            n_rotations=args.n_rotations or 2*ceil(log2(args.hidden_dim)),
                            activation=args.activation)
        return cell_
    else:
        return LSTMCell(args.hidden_dim)


for _ in range(args.n_layers - 1):
    rnn_out = RNN(cell(), return_sequences=True)(rnn_out)
    if args.use_ln:
        rnn_out = LayerNormalization()(rnn_out)
    if 0 < args.dropout < 1:
        rnn_out = Dropout(args.dropout,
                          noise_shape=(None, 1, args.hidden_dim))(rnn_out)
rnn_out = RNN(cell())(rnn_out)

if 0 < args.clf_dropout < 1:
    rnn_out = Dropout(args.clf_dropout)(rnn_out)

rnn_out = Dense(1, activation='sigmoid')(rnn_out)

model = Model(inp, rnn_out)

opt_cls = Adam if args.optimizer == 'adam' else RMSprop
model.compile(optimizer=opt_cls(learning_rate=args.lr, decay=args.decay),
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()
model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          callbacks=callbacks,
          validation_data=[x_val, y_val])

print('EVAL', model.evaluate(x_val, y_val))
if args.cell == 'vanilla' and args.no_validation:
    save_vprnn(model, args.model_name)
