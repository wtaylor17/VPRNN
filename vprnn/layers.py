from vpnn.layers import VPNNLayer
from . import base_layer
import keras.backend as K
from typing import List
from keras.models import Model
from math import log2, ceil


class VanillaCell(base_layer.BaseRNNCell):

    def __init__(self, units, use_diag=False, n_rotations=None, **kwargs):
        super(VanillaCell, self).__init__(units, **kwargs)
        if units % 2 != 0:
            raise ValueError('Need even no. of units')
        self.n_rotations = n_rotations or ceil(log2(units))
        self.recurrent_layer = None
        self.use_diag = use_diag
        self.use_kernel = True
        self.layer_config = {'units': units, 'n_rotations': n_rotations,
                             'use_diag': use_diag}

    def get_config(self):
        config = super(VanillaCell, self).get_config()
        config.update(self.layer_config)
        return config

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      constraint=self.kernel_constraint,
                                      regularizer=self.kernel_regularizer)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    constraint=self.bias_constraint,
                                    regularizer=self.bias_regularizer)
        if self.recurrent_layer is None:
            recurrent_layer = VPNNLayer(self.units,
                                             n_rotations=self.n_rotations,
                                             activation='linear',
                                             use_bias=False,
                                             name='vpnn',
                                             use_diag=self.use_diag,
                                             diag_func=None)  # TODO make diag_func a constructor parameter
            self.add_models([recurrent_layer])
        super(VanillaCell, self).build(input_shape)

    def call(self, inputs, states, **kwargs):
        x = inputs
        h_t_minus_1 = states[0]
        x_dot = K.dot(x, self.kernel)
        if self.use_kernel:
            h_dot = K.dot(h_t_minus_1, self.recurrent_kernel)
        else:
            h_dot = self.recurrent_layer(h_t_minus_1)
        h_t = self.activation(x_dot + h_dot + self.bias)
        return h_t, [h_t]

    def add_models(self, models: List[Model]):
        assert len(models) <= 1, "Zero or one model needs to be given"
        if self.recurrent_layer is not None:
            # stops weights from being double counted on a reload
            # (still shows up as non-trainable params)
            self.recurrent_layer.trainable = False
            del self.recurrent_layer
        if len(models) == 1:
            model = models[0]
            found_outp_size = K.int_shape(model.outputs[0])[-1]
            assert found_outp_size == self.units, f"Dims need to match, found {found_outp_size} != {self.units}"
            # its the recurrent_layer attribute, this assignment is sometimes redundant TODO fix that?
            self.recurrent_layer = model
            self.models = models
            self.recurrent_kernel = self.recurrent_layer(K.eye(self.units))
            if self.recurrent_clip is None:
                if self.time_steps is None:
                    self.recurrent_clip = -1
                else:
                    self.recurrent_clip = 2 ** (1 / self.time_steps)
            if self.recurrent_clip > 0:
                sgn = K.sign(self.recurrent_kernel)
                self.recurrent_kernel = K.clip(K.abs(self.recurrent_kernel), 0, self.recurrent_clip) * sgn
