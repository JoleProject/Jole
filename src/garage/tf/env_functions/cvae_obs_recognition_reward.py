"""Continuous MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.models.mlp_merge_twin_model import MLPMergeTwinModel
from garage.tf.env_functions.base2 import EnvFunction2


class CVAERewardRecognition(EnvFunction2):
    """
    Continuous MLP QFunction.

    This class implements a q value network to predict q based on the input
    state and action. It uses an MLP to fit the function of Q(s, a).

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Name of the q-function, also serves as the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        action_merge_layer (int): The index of layers at which to concatenate
            action inputs with the network. The indexing works like standard
            python list indexing. Index of 0 refers to the input layer
            (observation input) while an index of -1 points to the last
            hidden layer. Default parameter points to second layer from the
            end.
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
        input_include_goal (bool): Whether input includes goal.
        layer_normalization (bool): Bool for using layer normalization.

    """

    def __init__(self,
                 env_spec,
                 name='CVAERewardRecognition',
                 hidden_sizes=(32, 32),
                 action_merge_layer=-2,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 input_include_goal=False,
                 layer_normalization=False,
                 z_dim=5):
        super().__init__(name)

        self._env_spec = env_spec
        self._hidden_sizes = hidden_sizes
        self._action_merge_layer = action_merge_layer
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._input_include_goal = input_include_goal
        self._layer_normalization = layer_normalization
        self._z_dim = z_dim

        if self._input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ['observation', 'desired_goal'])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self.model = MLPMergeTwinModel(output_dim1=self._z_dim,
                                   output_dim2=self._z_dim,
                                   hidden_sizes=hidden_sizes,
                                   concat_layer=self._action_merge_layer,
                                   hidden_nonlinearity=hidden_nonlinearity,
                                   hidden_w_init=hidden_w_init,
                                   hidden_b_init=hidden_b_init,
                                   output_nonlinearity=output_nonlinearity,
                                   output_w_init=output_w_init,
                                   output_b_init=output_b_init,
                                   layer_normalization=layer_normalization)

        self._initialize()

    def _initialize(self):
        obs_ph = tf.compat.v1.placeholder(tf.float32, (None, self._obs_dim),
                                          name='obs')
        action_ph = tf.compat.v1.placeholder(tf.float32,
                                             (None, self._action_dim),
                                             name='act')
        reward = tf.compat.v1.placeholder(tf.float32,
          (None, 1), name="reward")

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(obs_ph, action_ph, reward)

        self._f_qval = tf.compat.v1.get_default_session().make_callable(
            self.model.networks['default'].outputs,
            feed_list=[obs_ph, action_ph, reward])

    def get_obs_val(self, observation, action, reward):
        """Q Value of the network."""
        return self._f_qval(observation, action, reward)

    @property
    def inputs(self):
        """Return the input tensor."""
        return self.model.networks['default'].inputs

    @overrides
    def get_fval_sym(self, state_input, action_input, reward, name):
        """
        Symbolic graph for q-network.

        Args:
            state_input (tf.Tensor): The state input tf.Tensor to the network.
            name (str): Network variable scope.

        Return:
            The tf.Tensor output of Discrete MLP QFunction.
        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            return self.model.build(state_input, action_input, reward, name=name)

    def clone(self, name):
        """
        Return a clone of the Q-function.

        It only copies the configuration of the Q-function,
        not the parameters.

        Args:
            name (str): Name of the newly created q-function.
        """
        return self.__class__(name=name,
                              env_spec=self._env_spec,
                              hidden_sizes=self._hidden_sizes,
                              action_merge_layer=self._action_merge_layer,
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init,
                              layer_normalization=self._layer_normalization,
                              z_dim=self._z_dim)

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_qval']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
