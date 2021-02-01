# reproduce, 8 has no real significance
from numpy.random import seed

seed(8)
try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import set_random_seed as set_seed
set_seed(8)

from random import seed

seed(8)

from vpnn.utils import adding_problem_generator
from keras.layers import Input, RNN, Dense, LSTMCell
from keras.models import Model, load_model
from keras import optimizers
from datetime import datetime
from keras.callbacks import TensorBoard

import os
import json
import sys
import argparse

sys.path.append('../../')
from vprnn.layers import VanillaCell
from vprnn.models import save_vprnn, load_vprnn

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--stamp', type=str,
                    default=str(datetime.now()).replace(' ', '-').replace(':', '.'))
parser.add_argument('--lr', type=float,
                    default=0.002)
parser.add_argument('--model', type=str,
                    default='vanilla')
parser.add_argument('--tensorboard',
                    action='store_true')
parser.add_argument('--noclip',
                    action='store_true')
parser.add_argument('--nodecay',
                    action='store_true')
parser.add_argument('--epochs',
                    type=int,
                    default=200)
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument('--rotations',
                    type=int,
                    default=7)
parser.add_argument('--dim',
                    type=int,
                    default=128)
parser.add_argument('--nodiag',
                    action='store_true')

args = parser.parse_args()
stamp = args.stamp
T = args.steps
H = args.dim
K = args.rotations
epochs = args.epochs
batch_size = args.batch_size
steps_per_epoch = 100
recurrent_activation = 'relu'
recurrent_clip = 2 ** (1 / T)
output_activation = 'linear'
model_type = args.model
initial_lr = args.lr
decay = initial_lr / (epochs * steps_per_epoch)
optimizer = optimizers.RMSprop(lr=initial_lr, decay=decay)
tensorboard = args.tensorboard

if args.noclip:
    recurrent_clip = -1

if args.nodecay:
    decay = 0

input_tensor = Input((T, 2))
if model_type == 'lstm':
    rnn_out = RNN(LSTMCell(H))(input_tensor)
    optimizers.Adam(lr=initial_lr, decay=decay, amsgrad=True)
else:
    cls = VanillaCell
    rnn_1 = RNN(cls(H, n_rotations=K,
                    activation=recurrent_activation,
                    recurrent_clip=recurrent_clip,
                    use_diag=not args.nodiag),
                return_sequences=True)
    rnn_2 = RNN(cls(H, n_rotations=K,
                    activation=recurrent_activation,
                    recurrent_clip=recurrent_clip,
                    use_diag=not args.nodiag))
    rnn_out = rnn_2(rnn_1(input_tensor))
dense = Dense(1, activation='linear')

model = Model(input_tensor, dense(rnn_out))
model.compile(loss='mse', optimizer=optimizer)
model.summary()

callbacks = [TensorBoard(log_dir=f'tblogs/{model_type}/{T}/{stamp}')] if tensorboard else None
data_generator_factory = adding_problem_generator(batch_size=batch_size, time_steps=T, center=True)
history = model.fit_generator(data_generator_factory(),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=data_generator_factory(),
                              validation_steps=10,
                              callbacks=callbacks).history

if not os.path.exists(f'models/{model_type}/{T}/{stamp}'):
    os.makedirs(f'models/{model_type}/{T}/{stamp}')

save_fn = save_vprnn if model_type != 'lstm' else (lambda mdl, path: mdl.save(os.path.join(path, 'model.h5')))
save_fn(model, f'models/{model_type}/{T}/{stamp}')

with open(f'models/{model_type}/{T}/{stamp}/run_config.json', 'w') as json_fp:
    json.dump({
        'K': K,
        'T': T,
        'H': H,
        'batch_size': batch_size,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'recurrent_activation': recurrent_activation,
        'output_activation': output_activation,
        'initial_lr': initial_lr,
        'decay': decay,
        'optimizer': type(optimizer).__name__,
        # 'amsgrad': True,
        'tensorboard': tensorboard,
        'recurrent_clip': recurrent_clip,
        'model_type': model_type
    }, json_fp, indent=4)

load_fn = (lambda path, **kwargs: load_vprnn(path, **kwargs)) if model_type != 'lstm' \
    else (lambda p, **kw: load_model(os.path.join(p, 'model.h5')))
model2 = load_fn(f'models/{model_type}/{T}/{stamp}', do_compile=False)
model2.compile(optimizer='sgd', loss='mse')

x, y = next(data_generator_factory())
print(f'LOADED MSE: {model2.evaluate(x, y)}')
print(f'MSE: {model.evaluate(x, y)}')
