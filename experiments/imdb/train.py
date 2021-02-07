from vprnn.layers import VanillaCell
from keras.layers import RNN, Embedding, Dense, Input, Dropout, LSTMCell
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras_layer_normalization import LayerNormalization
import argparse
from vprnn.models import save_vprnn
from vprnn.imdb_data import load_imdb_stash
from vprnn.imdb_data import create_embeddings_matrix
from math import log2, ceil


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=100)
parser.add_argument('--n_rotations', type=int, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--embedding_dropout', type=float, default=0.0)
parser.add_argument('--use_ln', action='store_true')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--model_name', type=str, default='imdb_model')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_words', type=int, default=25000)
parser.add_argument('--optimizer', type=str, default='rmsprop')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cell', type=str, default='vanilla')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--freeze_embeddings', action='store_true')
parser.add_argument('--test_mode', action='store_true')

args = parser.parse_args()
n_rotations = args.n_rotations or ceil(log2(args.hidden_dim))

(x_train, y_train), (x_test, y_test) = load_imdb_stash()
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

if args.test_mode:
    x_val, y_val = x_test, y_test
else:
    x_train, x_val = x_train[:-3000], x_train[-3000:]
    y_train, y_val = y_train[:-3000], y_train[-3000:]

embedding_mat, *_ = create_embeddings_matrix(dim=args.embedding_dim,
                                             num_words=args.num_words)
inp = Input((None,))
rnn_out = Embedding(args.num_words, args.embedding_dim,
                    embeddings_initializer=lambda *args_, **kwargs: K.constant(embedding_mat),
                    mask_zero=True,
                    trainable=not args.freeze_embeddings)(inp)

if 0 < args.embedding_dropout < 1:
    rnn_out = Dropout(args.embedding_dropout,
                      noise_shape=(None, 1, args.embedding_dim))(rnn_out)


def cell():
    if args.cell == 'vanilla':
        return VanillaCell(args.hidden_dim,
                           use_diag=False,
                           n_rotations=n_rotations,
                           activation=args.activation)
    else:
        return LSTMCell(args.hidden_dim)


for _ in range(args.n_layers-1):
    rnn_out = RNN(cell(), return_sequences=True)(rnn_out)
    if args.use_ln:
        rnn_out = LayerNormalization()(rnn_out)
    if 0 < args.dropout < 1:
        rnn_out = Dropout(args.dropout,
                          noise_shape=(None, 1, args.hidden_dim))(rnn_out)
rnn_out = RNN(cell())(rnn_out)
rnn_out = Dense(1, activation='sigmoid')(rnn_out)

model = Model(inp, rnn_out)

opt_cls = Adam if args.optimizer == 'adam' else RMSprop
model.compile(optimizer=opt_cls(learning_rate=args.lr),
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

if args.test_mode:
    def scheduler(epoch, _):
        return args.lr / (10 ** (epoch // 100))
    callbacks = [LearningRateScheduler(scheduler)]
else:
    callbacks = [EarlyStopping(monitor='val_acc',
                               mode='max',
                               restore_best_weights=True,
                               patience=args.patience)]

model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          callbacks=callbacks,
          verbose=2,
          validation_data=[x_val, y_val])

# if test_mode is true this is a test evaluation
print('EVALUATION', model.evaluate(x_val, y_val))
if args.cell == 'vanilla' and args.test_mode:
    save_vprnn(model, args.model_name)
