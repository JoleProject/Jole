"""
MLP Merge Model.

A model composed only of a multi-layer perceptron (MLP), which maps
real-valued inputs to real-valued outputs. This model is called an
MLP Merge Model because it takes two inputs and concatenates the second
input with the layer at a specified index. It can be merged with any layer
from the input layer to the last hidden layer.
"""
import tensorflow as tf

from garage.tf.core.twin_mlp import twin_mlp
from garage.tf.models.base import Model


class MLPMergeTwinModel(Model):
    """
    MLP Merge Model.

    Args:
        output_dim (int): Dimension of the network output.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        concat_layer (int): The index of layers at which to concatenate
            input_var2 with the network. The indexing works like standard
            python list indexing. Index of 0 refers to the input layer
            (input_var1) while an index of -1 points to the last hidden
            layer. Default parameter points to second layer from the end.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 output_dim1,
                 output_dim2,
                 name='MLPMergeModel',
                 hidden_sizes=(32, 32),
                 concat_layer=-2,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False,
                 with_sigma=True):
        super().__init__(name)
        self._output_dim1 = output_dim1
        self._output_dim2 = output_dim2
        self._hidden_sizes = hidden_sizes
        self._concat_layer = concat_layer
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._with_sigma = with_sigma

    def network_input_spec(self):
        """Network input spec."""
        return ['input_var1', 'input_var2', 'input_var3']

    def network_output_spec(self):
        """Network input spec."""
        if self._with_sigma:
            return ['output_var1', 'output_var2']
        return ['output_var1']

    def _build(self, state_input, action_input, z_input, name=None):
        return twin_mlp(
            input_var=state_input,
            output_dim1=self._output_dim1,
            output_dim2=self._output_dim2,
            hidden_sizes=self._hidden_sizes,
            input_var2=action_input,
            input_var3=z_input,
            concat_layer=self._concat_layer,
            name='mlp_concat',
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization,
            with_sigma=self._with_sigma)
