from .base_layer import BaseRNNCell
from .layers import VanillaCell


def custom_objects():
    global _custom_objects
    return _custom_objects


from . import mnist_data, base_layer, layers, models, utils, imdb_data
from vpnn import custom_objects as base_custom_object_fn


_custom_objects = {
    layer_cls.__name__: layer_cls
    for layer_cls in {BaseRNNCell, VanillaCell}
}
_custom_objects.update(base_custom_object_fn())
