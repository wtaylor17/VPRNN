import vpnn
from keras.layers import Layer
from abc import abstractmethod


class BaseRNNCell(Layer):
    def __init__(self, units,
                 recurrent_clip=-1,
                 activation='cheby',
                 models=None,
                 cheby_M=2.0,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 time_steps=None,
                 recurrent_dropout=0.,
                 **kwargs):
        """
        base class for any and all RNN cell classes in the vpnn package
        :param units: int, number of hidden units
        :param recurrent_clip: float, clip for max abs of recurrent kernel (None=>default to 2^(1/T), -1=>no clip)
        :param activation: keras Layer, str, or callable, the activation function to use
        :param cheby_M: float, chebyshev M value (used only if activation='cheby')
        :param use_bias: bool, if true a bias is used on the hidden state
        :param kernel_initializer: str or keras initializer for the non-recurrent kernel
        :param recurrent_initializer: str or keras initializer for the recurrent kernel
        :param bias_initializer: str or keras initializer for bias vector
        :param kernel_regularizer: str or keras regularizer for non-recurrent kernel
        :param recurrent_regularizer: str or keras regularizer for recurrent kernel
        :param bias_regularizer: str or keras regularizer for bias vector
        :param kernel_constraint: constraint for non-recurrent kernel
        :param recurrent_constraint: constraint for recurrent kernel
        :param bias_constraint: constraint for bias vector
        :param dropout: float, dropout level to apply to non-recurrent kernel during training
        :param time_steps: int or None, the sequence length (T)
        :param recurrent_dropout: float, dropout level to apply to recurrent kernel during training
        :param kwargs: passed to super constructor
        """
        self.units = units
        self.state_size = self.units
        self.time_steps = time_steps
        self.recurrent_clip = recurrent_clip
        self.activation = vpnn.utils.get_activation(activation, dim=self.units, cheby_M=cheby_M)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bias = None
        self.kernel = None
        self.recurrent_kernel = None
        self.recurrent_params = None
        self.built = False
        self.models = models or []
        self.base_layer_config = {
            'units': units,
            'recurrent_clip': recurrent_clip,
            'activation': activation,
            'cheby_M': cheby_M,
            'use_bias': use_bias,
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer,
            'kernel_constraint': kernel_constraint,
            'recurrent_initializer': recurrent_initializer,
            'recurrent_regularizer': recurrent_regularizer,
            'recurrent_constraint': recurrent_constraint,
            'bias_initializer': bias_initializer,
            'bias_regularizer': bias_regularizer,
            'bias_constraint': bias_constraint,
            'dropout': dropout,
            'recurrent_dropout': recurrent_dropout,
            'time_steps': time_steps
        }
        super(BaseRNNCell, self).__init__(**kwargs)

    @abstractmethod
    def add_models(self, models):
        raise NotImplementedError('Base layer - this function should be implemented to add all of the trained model'
                                  ' functionality to this model. E.g. The recurrent kernel is actually a VPNN Model.')

    def get_config(self):
        config = super(BaseRNNCell, self).get_config()
        config.update(self.base_layer_config)
        return config

    def build(self, input_shape):
        """
        build the model
        :param input_shape: tuple/tensor, input shape to the model
        """
        self.built = True
        super(BaseRNNCell, self).build(input_shape)

    def __call__(self, *args, **kwargs):
        """
        wrapper for call()
        """
        return super(BaseRNNCell, self).__call__(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        call the model: will raise an exception because this is not meant to be an instantiated class
        """
        raise NotImplementedError("RNNCell cannot be called, is meant to be a base class.")
