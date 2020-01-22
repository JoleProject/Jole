"""Deep Deterministic Policy Gradient (DDPG) implementation in TensorFlow."""
from collections import deque

from dowel import logger, tabular
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import random
from garage.misc.overrides import overrides
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class DynaDDPG(OffPolicyRLAlgorithm):
    """A DDPG model based on https://arxiv.org/pdf/1509.02971.pdf.

    DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
    method to optimize the policy and reward prediction. It uses a supervised
    method to update the critic network and policy gradient to update the actor
    network. And there are exploration strategy, replay buffer and target
    networks involved to stabilize the training process.

    Example:
        $ python garage/examples/tf/ddpg_pendulum.py

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        qf (object): The q value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        n_train_steps (int): Training steps.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        exploration_strategy (garage.np.exploration_strategies.
            ExplorationStrategy): Exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        policy_lr (float): Learning rate for training policy network.
        qf_lr (float): Learning rate for training q value network.
        discount(float): Discount factor for the cumulative return.
        policy_weight_decay (float): L2 weight decay factor for parameters
            of the policy network.
        qf_weight_decay (float): L2 weight decay factor for parameters
            of the q value network.
        policy_optimizer (tf.Optimizer): Optimizer for training policy network.
        qf_optimizer (tf.Optimizer): Optimizer for training q function
            network.
        clip_pos_returns (bool): Whether or not clip positive returns.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        max_action (float): Maximum action magnitude.
        reward_scale (float): Reward scale.
        input_include_goal (bool): Whether input includes goal.
        smooth_return (bool): Whether to smooth the return.
        name (str): Name of the algorithm shown in computation graph.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 obs_model,
                 reward_model,
                 replay_buffer,
                 ima_replay_buffer,
                 n_epoch_cycles=20,
                 n_train_steps=50,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 rollout_batch_size=1,
                 exploration_strategy=None,
                 target_update_tau=0.01,
                 policy_lr=1e-4,
                 qf_lr=1e-3,
                 obs_model_lr=1e-3,
                 reward_model_lr=1e-3,
                 jole_lr=1e-4,
                 discount=0.99,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                 qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                 obs_model_optimizer=tf.compat.v1.train.AdamOptimizer,
                 reward_model_optimizer=tf.compat.v1.train.AdamOptimizer,
                 jole_optimizer=tf.compat.v1.train.AdamOptimizer,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 max_action=None,
                 reward_scale=1.,
                 input_include_goal=False,
                 smooth_return=False,
                 name='DynaDDPG'):
        action_bound = env_spec.action_space.high
        self.obs_model = obs_model
        self.reward_model = reward_model
        self.max_action = action_bound if max_action is None else max_action
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.obs_model_lr = obs_model_lr
        self.reward_model_lr = reward_model_lr
        self.jole_lr = jole_lr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.policy_optimizer = policy_optimizer
        self.qf_optimizer = qf_optimizer
        self.obs_model_optimizer = obs_model_optimizer
        self.reward_model_optimizer = reward_model_optimizer
        self.jole_optimizer = jole_optimizer
        self.name = name
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.success_history = deque(maxlen=100)

        self.episode_rewards = []
        self.episode_policy_losses = []
        self.episode_qf_losses = []
        self.epoch_ys = []
        self.epoch_qs = []

        self.obs_model_loss = []
        self.reward_model_loss = []
        self.jole_loss = []

        self.target_policy = policy.clone('target_policy')
        self.target_qf = qf.clone('target_qf')

        self.ima_replay_buffer = ima_replay_buffer

        super(DynaDDPG, self).__init__(env_spec=env_spec,
                                   policy=policy,
                                   qf=qf,
                                   n_train_steps=n_train_steps,
                                   n_epoch_cycles=n_epoch_cycles,
                                   max_path_length=max_path_length,
                                   buffer_batch_size=buffer_batch_size,
                                   min_buffer_size=min_buffer_size,
                                   rollout_batch_size=rollout_batch_size,
                                   exploration_strategy=exploration_strategy,
                                   replay_buffer=replay_buffer,
                                   use_target=True,
                                   discount=discount,
                                   reward_scale=reward_scale,
                                   input_include_goal=input_include_goal,
                                   smooth_return=smooth_return)

    @overrides
    def init_opt(self):
        """Build the loss function and init the optimizer."""
        with tf.name_scope(self.name, 'DynaDDPG'):
            # Create target policy and qf network
            self.target_policy_f_prob_online = tensor_utils.compile_function(
                inputs=[self.target_policy.model.networks['default'].input],
                outputs=self.target_policy.model.networks['default'].outputs)
            self.target_qf_f_prob_online = tensor_utils.compile_function(
                inputs=self.target_qf.model.networks['default'].inputs,
                outputs=self.target_qf.model.networks['default'].outputs)
            self.get_predicted_reward_online = tensor_utils.compile_function(
                inputs=self.reward_model.model.networks['default'].inputs,
                outputs=self.reward_model.model.networks['default'].outputs)
            self.get_predicted_next_obs_online = tensor_utils.compile_function(
                inputs=self.obs_model.model.networks['default'].inputs,
                outputs=self.obs_model.model.networks['default'].outputs)

            # Set up target init and update function
            with tf.name_scope('setup_target'):
                ops = tensor_utils.get_target_ops(
                    self.policy.get_global_vars(),
                    self.target_policy.get_global_vars(), self.tau)
                policy_init_ops, policy_update_ops = ops
                qf_init_ops, qf_update_ops = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self.target_qf.get_global_vars(), self.tau)
                target_init_op = policy_init_ops + qf_init_ops
                target_update_op = policy_update_ops + qf_update_ops

            f_init_target = tensor_utils.compile_function(
                inputs=[], outputs=target_init_op)
            f_update_target = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('inputs'):
                if self.input_include_goal:
                    obs_dim = self.env_spec.observation_space.\
                        flat_dim_with_keys(['observation', 'desired_goal'])
                else:
                    obs_dim = self.env_spec.observation_space.flat_dim
                y = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, 1),
                                             name='input_y')
                obs = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim),
                                               name='input_observation')
                actions = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self.env_spec.action_space.flat_dim),
                    name='input_action')
                next_obs = tf.compat.v1.placeholder(
                    tf.float32, shape=(None, obs_dim), name='next_observation')
                reward = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, 1),
                                             name='reward')

            # Set up policy training function
            next_action = self.policy.get_action_sym(obs, name='policy_action')
            next_qval = self.qf.get_qval_sym(obs,
                                             next_action,
                                             name='policy_action_qval')
            with tf.name_scope('action_loss'):
                action_loss = -tf.reduce_mean(next_qval)
                if self.policy_weight_decay > 0.:
                    policy_reg = tc.layers.apply_regulaization(
                        tc.layers.l2_regularizer(self.policy_weight_decay),
                        weights_list=self.policy.get_regularizable_vars())
                    action_loss += policy_reg

            with tf.name_scope('minimize_action_loss'):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name='PolicyOptimizer').minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            f_train_policy = tensor_utils.compile_function(
                inputs=[obs], outputs=[policy_train_op, action_loss])

            # Set up qf training function
            qval = self.qf.get_qval_sym(obs, actions, name='q_value')
            with tf.name_scope('qval_loss'):
                qval_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(y, qval))
                if self.qf_weight_decay > 0.:
                    qf_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.qf_weight_decay),
                        weights_list=self.qf.get_regularizable_vars())
                    qval_loss += qf_reg

            with tf.name_scope('minimize_qf_loss'):
                qf_train_op = self.qf_optimizer(
                    self.qf_lr, name='QFunctionOptimizer').minimize(
                        qval_loss, var_list=self.qf.get_trainable_vars())

            f_train_qf = tensor_utils.compile_function(
                inputs=[y, obs, actions],
                outputs=[qf_train_op, qval_loss, qval])

            #Set up of environment model training function
            predicted_next_obs = self.obs_model.get_fval_sym(obs, actions, name='obs_value')
            predicted_reward = self.reward_model.get_fval_sym(obs, actions, name='reward_value')
            with tf.name_scope('model_loss'):
                obs_model_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(next_obs, predicted_next_obs))
                reward_model_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(reward, predicted_reward))

            with tf.name_scope('minimize_obs_model_loss'):
                obs_train_op = self.obs_model_optimizer(
                    self.obs_model_lr, name='ObsModelOptimizer').minimize(
                    obs_model_loss, var_list=self.obs_model.get_trainable_vars())
                reward_train_op = self.reward_model_optimizer(
                    self.reward_model_lr, name='RewardModelOptimizer').minimize(
                    reward_model_loss, var_list=self.reward_model.get_trainable_vars())

            f_train_obs_model = tensor_utils.compile_function(
                inputs=[next_obs, obs, actions],
                outputs=[obs_train_op, obs_model_loss])
            f_train_reward_model = tensor_utils.compile_function(
                inputs=[reward, obs, actions],
                outputs=[reward_train_op, reward_model_loss])

            self.f_train_reward_model = f_train_reward_model
            self.f_train_obs_model = f_train_obs_model
          
            self.f_train_policy = f_train_policy
            self.f_train_qf = f_train_qf
            self.f_init_target = f_init_target
            self.f_update_target = f_update_target

    def __getstate__(self):
        """Object.__getstate__."""
        data = self.__dict__.copy()
        del data['target_policy_f_prob_online']
        del data['target_qf_f_prob_online']
        del data['f_train_policy']
        del data['f_train_qf']
        del data['f_init_target']
        del data['f_update_target']
        return data

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__ = state
        self.init_opt()

    def train_once(self, itr, paths, obs_upper, obs_lower, action_upper, action_lower, evaluate_path):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        if itr>600:
            self.model_based_sampling(itr)
        paths = self.process_samples(itr, paths)

        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend([
            path for path, complete in zip(paths['undiscounted_returns'],
                                           paths['complete']) if complete
        ])
        self.success_history.extend([
            path for path, complete in zip(paths['success_history'],
                                           paths['complete']) if complete
        ])

        last_average_return = np.mean(self.episode_rewards)
        self.log_diagnostics(paths)
        for train_itr in range(self.n_train_steps):
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                self.evaluate = True
                qf_loss, y, q, policy_loss = self.optimize_policy(itr, paths)
                obs_model_loss, reward_model_loss = self.optimize_model(itr, paths)
                

                self.episode_policy_losses.append(policy_loss)
                self.episode_qf_losses.append(qf_loss)
                self.epoch_ys.extend(y)
                self.epoch_qs.extend(q)
                self.reward_model_loss.append(reward_model_loss)
                self.obs_model_loss.append(obs_model_loss)

        if itr % self.n_epoch_cycles == 0:
            logger.log('Training finished')

            if self.evaluate:
                tabular.record('Epoch', epoch)
                tabular.record('AverageReturn', np.mean(self.episode_rewards))
                tabular.record('StdReturn', np.std(self.episode_rewards))
                tabular.record('Policy/AveragePolicyLoss',
                               np.mean(self.episode_policy_losses))
                tabular.record('QFunction/AverageQFunctionLoss',
                               np.mean(self.episode_qf_losses))
                tabular.record('QFunction/AverageQ', np.mean(self.epoch_qs))
                tabular.record('QFunction/MaxQ', np.max(self.epoch_qs))
                tabular.record('QFunction/AverageAbsQ',
                               np.mean(np.abs(self.epoch_qs)))
                tabular.record('QFunction/AverageY', np.mean(self.epoch_ys))
                tabular.record('QFunction/MaxY', np.max(self.epoch_ys))
                tabular.record('QFunction/AverageAbsY',
                               np.mean(np.abs(self.epoch_ys)))
                tabular.record('Jole/ObsModelLoss',
                               np.mean(self.obs_model_loss))
                tabular.record('Jole/RewardModelLoss',
                               np.mean(self.reward_model_loss))
                
                if self.input_include_goal:
                    tabular.record('AverageSuccessRate',
                                   np.mean(self.success_history))

            if not self.smooth_return:
                self.episode_rewards = []
                self.episode_policy_losses = []
                self.episode_qf_losses = []
                self.epoch_ys = []
                self.epoch_qs = []
                self.obs_model_loss = []
                self.reward_model_loss = []
                self.sepe_obs_model_loss = []
                self.sepe_reward_model_loss = []
                self.jole_loss = []

            self.success_history.clear()

        return last_average_return

    def optimize_model(self, itr, samples_data):
        """Perform model optimizing
        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        rewards = rewards.reshape(-1, 1)
        _, obs_model_loss = self.f_train_obs_model(next_observations, observations, actions)
        _, reward_model_loss = self.f_train_reward_model(rewards, observations, actions)

        return obs_model_loss, reward_model_loss

    def model_based_sampling(self, itr):
        transitions = self.replay_buffer.sample(5)
        observations = transitions['observation']
        for i, observation in enumerate(observations):
            obses = [observation]
            n_samples = 0
            while n_samples < 100: 
                #obs_normalized = tensor_utils.normalize_pixel_batch(
                #    self.env_spec, input_obses)
                if self.es:
                    actions, agent_infos = self.es.get_actions(
                        itr, obses, self.policy)
                else:
                    actions, agent_infos = self.policy.get_actions(
                        obses)

                rewards = self.get_predicted_reward_online(obses, actions)
                next_obses = self.get_predicted_next_obs_online(obses, actions)
                dones = [False] #to-do: update this function

                self.ima_replay_buffer.add_transitions(
                            observation=obses,
                            action=actions,
                            reward=rewards * self.reward_scale,
                            terminal=dones,
                            next_observation=next_obses,
                        )

                n_samples += 1
                obses = next_obses
                #print("Model-Based Sampling....Epoch:{}, Step:{}".format(i, n_samples))

    @overrides
    def optimize_policy(self, itr, samples_data):
        """Perform algorithm optimizing.

        Args:
            itr (int): Iterations.
            samples_data (list): Processed batch data.

        Returns:
            action_loss (float): Loss of action predicted by the policy network
            qval_loss (float): Loss of q value predicted by the q network.
            ys (float): y_s.
            qval (float): Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)

        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        
        if self.ima_replay_buffer.n_transitions_stored >= self.min_buffer_size: 
            ima_transitions = self.ima_replay_buffer.sample(self.buffer_batch_size)

            observations = np.concatenate((observations, ima_transitions['observation']), axis=0)
            rewards = np.concatenate((rewards, np.reshape(ima_transitions['reward'], [-1,])), axis=0)
            actions = np.concatenate((actions, ima_transitions['action']), axis=0)
            next_observations = np.concatenate((next_observations, ima_transitions['next_observation']), axis=0)
            terminals = np.concatenate((terminals, ima_transitions['terminal']), axis=0)
        

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        if self.input_include_goal:
            goals = transitions['goal']
            next_inputs = np.concatenate((next_observations, goals), axis=-1)
            inputs = np.concatenate((observations, goals), axis=-1)
        else:
            next_inputs = next_observations
            inputs = observations

        target_actions = self.target_policy_f_prob_online(next_inputs)
        target_qvals = self.target_qf_f_prob_online(next_inputs,
                                                    target_actions)

        clip_range = (-self.clip_return,
                      0. if self.clip_pos_returns else self.clip_return)
        ys = np.clip(
            rewards + (1.0 - terminals) * self.discount * target_qvals,
            clip_range[0], clip_range[1])

        _, qval_loss, qval = self.f_train_qf(ys, inputs, actions)
        _, action_loss = self.f_train_policy(inputs)

        self.f_update_target()

        return qval_loss, ys, qval, action_loss

    @overrides
    def get_itr_snapshot(self, itr):
        """Return data saved in the snapshot for this iteration."""
        return dict(itr=itr, policy=self.policy)
