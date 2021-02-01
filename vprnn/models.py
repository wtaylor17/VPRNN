from keras.models import Model, load_model
from . import base_layer
from keras.layers import Input
from keras.layers import RNN
from keras.layers import Bidirectional
from keras.utils.generic_utils import CustomObjectScope
from . import custom_objects
import os


def get_models_from_rnn_cells(model: Model):
    with CustomObjectScope(custom_objects()):
        models = []
        for layer in model.layers:
            if isinstance(layer, Bidirectional):
                # for bidirectional go forward then backward, treat like 2 RNNs
                rnn = layer.forward_layer
                if isinstance(rnn, RNN) and hasattr(rnn.cell, 'models'):
                    nested_rnn_cell_models = rnn.cell.models
                    models.append(nested_rnn_cell_models)
                rnn = layer.backward_layer
                if isinstance(rnn, RNN) and hasattr(rnn.cell, 'models'):
                    nested_rnn_cell_models = rnn.cell.models
                    models.append(nested_rnn_cell_models)
            elif isinstance(layer, RNN) and hasattr(layer.cell, 'models'):
                nested_rnn_cell_models = layer.cell.models
                models.append(nested_rnn_cell_models)
        return models


def save_vprnn(model: Model, path: str):
    with CustomObjectScope(custom_objects()):
        nested_models = get_models_from_rnn_cells(model)
        for i, rnn_model_list in enumerate(nested_models):
            if not os.path.exists(os.path.join(path, f'rnn{i}')):
                os.makedirs(os.path.join(path, f'rnn{i}'))
            for j, nested_model in enumerate(rnn_model_list):
                nested_model.save(os.path.join(path, f'rnn{i}', f'model{j}.h5'))
        model.save(os.path.join(path, 'main_model.h5'))
    return len(nested_models)


def load_vprnn(path: str, do_compile=True, **kwargs):
    assert os.path.isdir(path), "Needs to be a valid directory"
    c_o = custom_objects()
    with CustomObjectScope(c_o):
        n_rnns = 0
        model: Model = load_model(os.path.join(path, 'main_model.h5'),
                                  custom_objects=c_o, compile=do_compile)
        for layer in model.layers:
            if isinstance(layer, RNN) and isinstance(layer.cell, base_layer.BaseRNNCell):
                # open up path/rnn{i}
                models = []
                for model_path in sorted(os.listdir(f'{path}/rnn{n_rnns}')):
                    if model_path.endswith('.h5'):
                        models.append(load_model(f'{path}/rnn{n_rnns}/{model_path}',
                                                 custom_objects=custom_objects()))
                n_rnns += 1
                layer.cell.add_models(models)
            elif isinstance(layer, Bidirectional):
                for rnn in [layer.forward_layer, layer.backward_layer]:
                    if isinstance(rnn.cell, base_layer.BaseRNNCell):
                        models = []
                        for model_path in sorted(os.listdir(f'{path}/rnn{n_rnns}')):
                            if model_path.endswith('.h5'):
                                models.append(load_model(f'{path}/rnn{n_rnns}/{model_path}',
                                                         custom_objects=custom_objects()))
                        n_rnns += 1
                        rnn.cell.add_models(models)
        inp_layer = Input(model.input_shape[1:])
        outp = inp_layer
        for layer in model.layers[1:]:
            outp = layer(outp)
        new_model: Model = Model(inp_layer, outp, **kwargs)
        return new_model
