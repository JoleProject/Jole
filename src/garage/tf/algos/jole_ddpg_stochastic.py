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


class JoLeDDPGStochastic(OffPolicyRLAlgorithm):
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
                 obs_model_recognition,
                 obs_model_generator,
                 reward_model_recognition,
                 reward_model_generator,
                 replay_buffer,
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
                 name='JoLeDDPG',
                 z_dim=10):
        action_bound = env_spec.action_space.high
        self.obs_model_recognition = obs_model_recognition
        self.obs_model_generator = obs_model_generator
        self.reward_model_recognition = reward_model_recognition
        self.reward_model_generator = reward_model_generator
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
        self.epoch_jole_qval = []
        self.epoch_jole_ys = []
        self.epoch_jole_ys_before_clip = []

        self.obs_model_loss = []
        self.reward_model_loss = []
        self.sepe_obs_model_loss = []
        self.sepe_reward_model_loss = []
        self.jole_loss = []
        self.jole_loss_observed = []

        self.obs_model_reconstr_loss = []
        self.reward_model_reconstr_loss = []
        self.sepe_obs_model_reconstr_loss = []
        self.sepe_reward_model_reconstr_loss = []
        self.obs_recontr_mean = []
        self.obs_reconstr_log_sigma = []
        self.debug_obs_mse_pure = []
        self.debug_exp_sigma_part = []

        self.target_policy = policy.clone('target_policy')
        self.target_qf = qf.clone('target_qf')

        self.sepe_obs_model_generator = obs_model_generator.clone('sepe_obs_model_generator')
        self.sepe_obs_model_recognition = obs_model_recognition.clone('sepe_obs_model_recognition')
        self.sepe_reward_model_generator = reward_model_generator.clone('sepe_reward_model_generator')
        self.sepe_reward_model_recognition = reward_model_recognition.clone('sepe_reward_model_recognition')

        self.sepe_reward_model_win = 0
        self.sepe_obs_model_win = 0
        self.use_jole_qf = 0.0
        self.use_jole_obs = 0.0
        self.use_jole_reward = 0.0
        self.jole_clip_buffer = 10.0
        self.jole_clip_return_min = -np.inf
        self.jole_clip_return_max = np.inf
        self.dim_z = z_dim
        self.num_z = 10

        super(JoLeDDPGStochastic, self).__init__(env_spec=env_spec,
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
        with tf.name_scope(self.name, 'JoLeDDPG'):
            # Create target policy and qf network
            self.target_policy_f_prob_online = tensor_utils.compile_function(
                inputs=[self.target_policy.model.networks['default'].input],
                outputs=self.target_policy.model.networks['default'].outputs)
            self.target_qf_f_prob_online = tensor_utils.compile_function(
                inputs=self.target_qf.model.networks['default'].inputs,
                outputs=self.target_qf.model.networks['default'].outputs)

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

                jole_obs = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim),
                                               name='jole_input_observation')
                jole_actions = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self.env_spec.action_space.flat_dim),
                    name='jole_input_action')

                jole_clip_return_min = tf.compat.v1.placeholder(
                    tf.float32, shape=(), name="jole_clip_return_min")
                jole_clip_return_max = tf.compat.v1.placeholder(
                    tf.float32, shape=(), name="jole_clip_return_max")
                use_jole = tf.compat.v1.placeholder(
                    tf.float32, shape=(), name="use_jole")

                reguzs = tf.compat.v1.placeholder(
                    tf.float32, shape=(self.num_z, None, self.dim_z), name='reguzs')
                eps = tf.compat.v1.placeholder(
                    tf.float32, shape=(None, self.dim_z), name='eps')

            # Set up policy training function
            next_action = self.policy.get_action_sym(obs, name='policy_action')
            next_qval = self.qf.get_qval_sym(obs,
                                             next_action,
                                             name='policy_action_qval')
            with tf.name_scope('action_loss'):
                action_loss = -tf.reduce_mean(next_qval)
                if self.policy_weight_decay > 0.:
                    policy_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.policy_weight_decay),
                        weights_list=self.policy.get_regularizable_vars())
                    action_loss += policy_reg

            with tf.name_scope('minimize_action_loss'):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name='PolicyOptimizer').minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            f_train_policy = tensor_utils.compile_function(
                inputs=[obs], outputs=[policy_train_op, action_loss])

            #get jole loss
            jole_qval = self.qf.get_qval_sym(jole_obs, jole_actions, name='jole_q_value') #[batch_size]
            jole_ys_samples_before_clip = []
            jole_ys_samples = []
            for i in range(self.num_z):
                regu_z = reguzs[i]
                jole_predicted_next_obs, _ = self.obs_model_generator.get_fval_sym(jole_obs, jole_actions, regu_z, name="jole_obs_value{}"
                                                                                .format(i))
                jole_predicted_reward = self.reward_model_generator.get_fval_sym(jole_obs, jole_actions, regu_z, name="jole_reward{}".format(i));
                jole_predicted_next_action = self.target_policy.get_action_sym(jole_predicted_next_obs, name='jole_policy_action{}'.format(i))
                jole_ys_before_clip = jole_predicted_reward + self.discount*self.target_qf.get_qval_sym(jole_predicted_next_obs, jole_predicted_next_action, name="jole_ys_{}".format(i))
                jole_ys_sample = tf.clip_by_value(jole_ys_before_clip, jole_clip_return_min, jole_clip_return_max)
                jole_ys_samples_before_clip.append(jole_ys_before_clip)
                jole_ys_samples.append(jole_ys_sample)
            jole_ys = tf.reduce_mean(jole_ys_samples, axis=0)  #[batch_size]
            jole_ys_before_clip =tf.reduce_mean(jole_ys_samples_before_clip, axis=0)
            with tf.name_scope('jole_loss'):
                jole_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(jole_qval, jole_ys))

            f_cal_jole_loss = tensor_utils.compile_function(
                inputs=[jole_obs, jole_actions, jole_clip_return_min, jole_clip_return_max, use_jole, reguzs],
                outputs=[jole_loss, jole_qval, jole_ys, jole_ys_before_clip])

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
                qval_loss += use_jole * 0.2 * jole_loss

            with tf.name_scope('minimize_qf_loss'):
                qf_train_op = self.qf_optimizer(
                    self.qf_lr, name='QFunctionOptimizer').minimize(
                        qval_loss, var_list=self.qf.get_trainable_vars())

            f_train_qf = tensor_utils.compile_function(
                inputs=[y, obs, actions, jole_obs, jole_actions, jole_clip_return_min, jole_clip_return_max, use_jole, reguzs],
                outputs=[qf_train_op, qval_loss, qval])

            #obs cvae model
            z_obs_mean, z_obs_log_sigma = self.obs_model_recognition.get_fval_sym(obs, actions, next_obs, "obs_model_recoginition")
            z_obs_sample = z_obs_mean + tf.multiply(tf.sqrt(tf.exp(z_obs_log_sigma)), eps)
            obs_recontr_mean, obs_reconstr_log_sigma = self.obs_model_generator.get_fval_sym(obs, actions, z_obs_sample, "obs_model_generator")
            obs_variance = tf.exp(obs_reconstr_log_sigma)
            log2pi = tf.log(tf.sqrt(2*np.pi))
            reconstr_loss_obs = tf.reduce_mean((log2pi) + tf.reduce_sum(\
                tf.square(next_obs - obs_recontr_mean)/(2*obs_variance)+(0.05*obs_reconstr_log_sigma),1))
            latent_loss_obs =  tf.reduce_mean(-0.5*tf.reduce_sum(1. + z_obs_log_sigma-tf.square(z_obs_mean)-tf.exp(z_obs_log_sigma),1))
            cvae_cost_obs = reconstr_loss_obs + latent_loss_obs

            debug_obs_mse = tf.reduce_mean(tf.reduce_sum(tf.square(next_obs - obs_recontr_mean)/(2*obs_variance), 1))
            debug_obs_mse_pure = tf.reduce_mean(tf.reduce_sum(tf.square(next_obs - obs_recontr_mean), 1))
            debug_exp_sigma_part = tf.reduce_mean(tf.reduce_sum(obs_variance, 1))
            debug_sigma_part = tf.reduce_mean(tf.reduce_sum(0.5 * obs_reconstr_log_sigma, 1))

            #reward cvae model
            z_reward_mean, z_reward_log_sigma = self.reward_model_recognition.get_fval_sym(obs, actions, reward, "reward_model_recoginition")
            z_reward_sample = z_reward_mean + tf.multiply(tf.sqrt(tf.exp(z_reward_log_sigma)), eps)
            reward_recontr_mean, reward_reconstr_log_sigma = self.reward_model_generator.get_fval_sym(obs, actions, z_reward_sample, "reward_model_generator")
            reward_variance = tf.exp(reward_reconstr_log_sigma)
            reconstr_loss_reward = tf.reduce_mean((log2pi) + tf.reduce_sum(\
                tf.square(reward - reward_recontr_mean)/(2*reward_variance)+(0.05*reward_reconstr_log_sigma),1))
            latent_loss_reward =  tf.reduce_mean(-0.5*tf.reduce_sum(1. + z_reward_log_sigma-tf.square(z_reward_mean)-tf.exp(z_reward_log_sigma),1))
            cvae_cost_reward = reconstr_loss_reward + latent_loss_reward

            with tf.name_scope('model_loss'):
                #change to predict the delta of s
                obs_model_loss = cvae_cost_obs+ use_jole * 0.0001 * jole_loss
                reward_model_loss = cvae_cost_reward + use_jole * 0.000001 * jole_loss

            with tf.name_scope('minimize_obs_model_loss'):
                obs_train_op = self.obs_model_optimizer(
                    self.obs_model_lr, name='ObsModelOptimizer').minimize(
                    obs_model_loss, var_list=self.obs_model_generator.get_trainable_vars()+self.obs_model_recognition.get_trainable_vars())
                reward_train_op = self.reward_model_optimizer(
                    self.reward_model_lr, name='RewardModelOptimizer').minimize(
                    reward_model_loss, var_list=self.reward_model_generator.get_trainable_vars()+self.reward_model_recognition.get_trainable_vars())

            self.f_train_obs_model = tensor_utils.compile_function(
                inputs=[next_obs, obs, actions, jole_obs, jole_actions, jole_clip_return_min, jole_clip_return_max, use_jole, reguzs, eps],
                outputs=[obs_train_op, cvae_cost_obs, reconstr_loss_obs, debug_obs_mse, debug_sigma_part, obs_reconstr_log_sigma, obs_variance])
            self.f_train_reward_model = tensor_utils.compile_function(
                inputs=[reward, obs, actions, jole_obs, jole_actions, jole_clip_return_min, jole_clip_return_max, use_jole, reguzs, eps],
                outputs=[reward_train_op, cvae_cost_reward, reconstr_loss_reward])

            #Set up of seperate environment model training function
            #seperate obs cvae model
            sz_obs_mean, sz_obs_log_sigma = self.sepe_obs_model_recognition.get_fval_sym(obs, actions, next_obs, "obs_model_recoginition")
            sz_obs_sample = sz_obs_mean + tf.multiply(tf.sqrt(tf.exp(sz_obs_log_sigma)), eps)
            sobs_recontr_mean, sobs_reconstr_log_sigma = self.sepe_obs_model_generator.get_fval_sym(obs, actions, sz_obs_sample, "obs_model_generator")
            sobs_variance = tf.exp(sobs_reconstr_log_sigma)
            sreconstr_loss_obs = tf.reduce_mean((log2pi) + tf.reduce_sum(\
                tf.square(next_obs - sobs_recontr_mean)/(2*sobs_variance)+(0.05*sobs_reconstr_log_sigma),1))
            slatent_loss_obs =  tf.reduce_mean(-0.5*tf.reduce_sum(1. + sz_obs_log_sigma-tf.square(sz_obs_mean)-tf.exp(sz_obs_log_sigma),1))
            scvae_cost_obs = sreconstr_loss_obs + slatent_loss_obs

            #seperate reward cvae model
            sz_reward_mean, sz_reward_log_sigma = self.sepe_reward_model_recognition.get_fval_sym(obs, actions, reward, "reward_model_recoginition")
            sz_reward_sample = sz_reward_mean + tf.multiply(tf.sqrt(tf.exp(sz_reward_log_sigma)), eps)
            sreward_recontr_mean, sreward_reconstr_log_sigma = self.sepe_reward_model_generator.get_fval_sym(obs, actions, sz_reward_sample, "reward_model_generator")
            sreward_variance = tf.exp(sreward_reconstr_log_sigma)
            sreconstr_loss_reward = tf.reduce_mean((log2pi) + tf.reduce_sum(\
                tf.square(reward - sreward_recontr_mean)/(2*sreward_variance)+(0.05*sreward_reconstr_log_sigma),1))
            slatent_loss_reward =  tf.reduce_mean(-0.5*tf.reduce_sum(sz_reward_log_sigma-tf.square(sz_reward_mean)-tf.exp(sz_reward_log_sigma),1))
            scvae_cost_reward = sreconstr_loss_reward + slatent_loss_reward

            with tf.name_scope('seperate_model_loss'):
                #change to predict the delta of s
                sobs_model_loss = scvae_cost_obs
                sreward_model_loss = scvae_cost_reward

            with tf.name_scope('minimize_sepertate_model_loss'):
                sepe_obs_train_op = self.obs_model_optimizer(
                    self.obs_model_lr, name='SepeObsModelOptimizer').minimize(
                    sobs_model_loss, var_list=self.sepe_obs_model_generator.get_trainable_vars()+self.sepe_obs_model_recognition.get_trainable_vars())
                sepe_reward_train_op = self.reward_model_optimizer(
                    self.reward_model_lr, name='SepeRewardModelOptimizer').minimize(
                    sreward_model_loss, var_list=self.sepe_reward_model_generator.get_trainable_vars()+self.sepe_reward_model_recognition.get_trainable_vars())

            self.f_train_sepe_obs_model = tensor_utils.compile_function(
                inputs=[next_obs, obs, actions, eps],
                outputs=[sepe_obs_train_op, scvae_cost_obs, sreconstr_loss_obs])
            self.f_train_sepe_reward_model = tensor_utils.compile_function(
                inputs=[reward, obs, actions, eps],
                outputs=[sepe_reward_train_op, scvae_cost_reward, sreconstr_loss_reward])

            # Copy the parameter of seperate env models when necessary
            with tf.name_scope('copy_sepe_env_models'):
                copy_sepe_obs_model_recognition_ops = tensor_utils.get_target_ops(
                    self.sepe_obs_model_recognition.get_global_vars(),
                    self.obs_model_recognition.get_global_vars())

                copy_sepe_obs_model_generator_ops = tensor_utils.get_target_ops(
                    self.sepe_obs_model_generator.get_global_vars(),
                    self.obs_model_generator.get_global_vars())

                copy_sepe_reward_model_recognition_ops = tensor_utils.get_target_ops(
                    self.sepe_reward_model_recognition.get_global_vars(),
                    self.reward_model_recognition.get_global_vars())

                copy_sepe_reward_model_generator_ops = tensor_utils.get_target_ops(
                    self.sepe_reward_model_generator.get_global_vars(),
                    self.reward_model_generator.get_global_vars())

            f_copy_sepe_obs_model = tensor_utils.compile_function(
                inputs=[], outputs=[copy_sepe_obs_model_recognition_ops,copy_sepe_obs_model_generator_ops])
            f_copy_sepe_reward_model = tensor_utils.compile_function(
                inputs=[], outputs=[copy_sepe_reward_model_recognition_ops,copy_sepe_reward_model_generator_ops])

            self.f_copy_sepe_reward_model = f_copy_sepe_reward_model
            self.f_copy_sepe_obs_model = f_copy_sepe_obs_model

            self.f_cal_jole_loss = f_cal_jole_loss

            self.f_train_policy = f_train_policy
            self.f_train_qf = f_train_qf
            self.f_init_target = f_init_target
            self.f_update_target = f_update_target

            #self.f_obs_model_predict = f_obs_model_predict
            #self.f_reward_model_predict = f_reward_model_predict

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

    def train_once(self, itr, paths, obs_upper, obs_lower, action_upper, action_lower, evaluate_paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        #print("self.use_jole:", type(self.use_jole))
        paths = self.process_samples(itr, paths)
        evaluate_paths = self.process_samples(itr, evaluate_paths)

        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend([
            path for path in zip(evaluate_paths['undiscounted_returns'])
        ])

        #self.episode_rewards.extend([
        #    path for path, complete in zip(paths['undiscounted_returns'],
        #                                   paths['complete']) if complete
        #])
        self.success_history.extend([
            path for path, complete in zip(paths['success_history'],
                                           paths['complete']) if complete
        ])

        last_average_return = np.mean(self.episode_rewards)

        if itr%self.n_epoch_cycles == 0:
            if itr>600:
                self.use_jole_qf = 1.0
                self.use_jole_obs = 1.0
                self.use_jole_reward = 1.0
            else:
                self.use_jole_qf = 0.0
                self.use_jole_obs = 0.0
                self.use_jole_reward = 0.0

        self.log_diagnostics(paths)
        for train_itr in range(self.n_train_steps):
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                self.evaluate = True
                transitions, observations, rewards, actions, next_observations, terminals = self.sample_transitions(itr, paths)
                jole_loss, jole_loss_observed, jole_qval, jole_ys, jole_ys_before_clip, jole_obs, jole_actions, regu_zs = self.get_jole_obs_actions(itr, observations, actions, obs_upper, obs_lower, action_upper, action_lower, train_jole = True, jole_obs_action_type = "policy_sample")
                qf_loss, y, q, policy_loss = self.optimize_policy(transitions, observations, rewards, actions, next_observations, terminals, jole_obs, jole_actions, regu_zs)
                obs_model_loss, reward_model_loss, sepe_obs_model_loss, sepe_reward_model_loss,\
                    obs_model_reconstr_loss, reward_model_reconstr_loss, sepe_obs_model_reconstr_loss, sepe_reward_model_reconstr_loss,\
                    obs_recontr_mean, obs_reconstr_log_sigma, obs_reconstr_log_sigma, obs_variance = self.optimize_model(itr, paths, jole_obs, jole_actions, regu_zs)
                #print(obs_reconstr_log_sigma[1,:])
                #print(obs_variance[1,:])
                #jole_loss = self.optimize_jole(itr, paths, obs_upper, obs_lower, action_upper, action_lower, train_jole = True, jole_obs_action_type = "random_sample")

                self.episode_policy_losses.append(policy_loss)
                self.episode_qf_losses.append(qf_loss)
                self.epoch_ys.append(y)
                self.epoch_qs.append(q)
                self.epoch_jole_ys.append(jole_ys)
                self.epoch_jole_ys_before_clip.append(jole_ys_before_clip)
                self.epoch_jole_qval.append(jole_qval)
                self.reward_model_loss.append(reward_model_loss)
                self.obs_model_loss.append(obs_model_loss)
                self.sepe_reward_model_loss.append(sepe_reward_model_loss)
                self.sepe_obs_model_loss.append(sepe_obs_model_loss)
                self.obs_model_reconstr_loss.append(obs_model_reconstr_loss)
                self.reward_model_reconstr_loss.append(reward_model_reconstr_loss)
                self.sepe_obs_model_reconstr_loss.append(sepe_obs_model_reconstr_loss)
                self.sepe_reward_model_reconstr_loss.append(sepe_reward_model_reconstr_loss)
                self.jole_loss.append(jole_loss)
                self.jole_loss_observed.append(jole_loss_observed)
                self.obs_recontr_mean.append(obs_recontr_mean)
                self.obs_reconstr_log_sigma.append(obs_reconstr_log_sigma)

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
                tabular.record('JoLe/AverageJoleQ', np.mean(self.epoch_jole_qval))
                tabular.record('JoLe/MaxJoleQ', np.max(self.epoch_jole_qval))
                tabular.record('JoLe/AverageAbsJoleQ',
                               np.mean(np.abs(self.epoch_jole_qval)))
                tabular.record('JoLe/AverageJoleY', np.mean(self.epoch_jole_ys))
                tabular.record('JoLe/MaxJoleY', np.max(self.epoch_jole_ys))
                tabular.record('JoLe/AverageAbsJoleY',
                               np.mean(np.abs(self.epoch_jole_ys)))
                tabular.record('JoLe/AverageJoleYBeforeClip', np.mean(self.epoch_jole_ys_before_clip))
                tabular.record('JoLe/MaxJoleYBeforeClip', np.max(self.epoch_jole_ys_before_clip))
                tabular.record('JoLe/AverageAbsJoleYBeforeClip',
                               np.mean(np.abs(self.epoch_jole_ys_before_clip)))
                tabular.record('Jole/ObsModelLoss',
                               np.mean(self.obs_model_loss))
                tabular.record('Jole/RewardModelLoss',
                               np.mean(self.reward_model_loss))
                tabular.record('Jole/SepeObsModelLoss',
                               np.mean(self.sepe_obs_model_loss))
                tabular.record('Jole/SepeRewardModelLoss',
                               np.mean(self.sepe_reward_model_loss))
                tabular.record('Jole/ObsModelReconstrLoss',
                               np.mean(self.obs_model_reconstr_loss))
                tabular.record('Jole/obs_recontr_mean',
                               np.mean(self.obs_recontr_mean))
                tabular.record('Jole/obs_reconstr_log_sigma',
                               np.mean(self.obs_reconstr_log_sigma))
                tabular.record('Jole/RewardModelReconstrLoss',
                               np.mean(self.reward_model_reconstr_loss))
                tabular.record('Jole/SepeObsModelReconstrLoss',
                               np.mean(self.sepe_obs_model_reconstr_loss))
                tabular.record('Jole/SepeRewardModelReconstrLoss',
                               np.mean(self.sepe_reward_model_reconstr_loss))
                tabular.record('Jole/JoleLoss',
                               np.mean(self.jole_loss))
                tabular.record('Jole/JoleLossObserved',
                               np.mean(self.jole_loss_observed))
                
                if self.input_include_goal:
                    tabular.record('AverageSuccessRate',
                                   np.mean(self.success_history))
            if len(self.epoch_ys) > 0:
                self.jole_clip_return_min = float(np.min(self.epoch_ys) - self.jole_clip_buffer)
                #print("self.jole_clip_return_min:", type(self.jole_clip_return_min))
                self.jole_clip_return_max = float(np.max(self.epoch_ys) + self.jole_clip_buffer)
                print("scope:", self.jole_clip_return_min, self.jole_clip_return_max)
                #print("self.jole_clip_return_max:", self.jole_clip_return_max)

            if not self.smooth_return:
                self.episode_rewards = []
                self.episode_policy_losses = []
                self.episode_qf_losses = []
                self.epoch_ys = []
                self.epoch_qs = []
                self.epoch_jole_ys = []
                self.epoch_jole_qval = []
                self.epoch_jole_ys_before_clip = []
                self.obs_model_loss = []
                self.reward_model_loss = []
                self.sepe_obs_model_loss = []
                self.sepe_reward_model_loss = []
                self.jole_loss = []
                self.jole_loss_observed = []
                self.obs_model_reconstr_loss = []
                self.reward_model_reconstr_loss = []
                self.sepe_obs_model_reconstr_loss = []
                self.sepe_reward_model_reconstr_loss = []
                self.obs_recontr_mean = []
                self.obs_reconstr_log_sigma = []
                self.debug_obs_mse_pure = []
                self.debug_exp_sigma_part = []

            self.success_history.clear()

        return last_average_return

    def optimize_model(self, itr, samples_data, jole_obs, jole_actions, regu_zs):
        """Perform model optimizing
        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        rewards = rewards.reshape(-1, 1)
        eps = np.random.normal(0, 1, (self.buffer_batch_size, self.dim_z))
        _, obs_model_loss, obs_model_reconstr_loss, obs_recontr_mean, obs_reconstr_log_sigma, obs_reconstr_log_sigma, obs_variance = self.f_train_obs_model(next_observations, observations, actions, jole_obs, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, self.use_jole_obs, regu_zs, eps)
        _, reward_model_loss, reward_model_reconstr_loss = self.f_train_reward_model(rewards, observations, actions, jole_obs, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, self.use_jole_reward, regu_zs, eps)

        _, sepe_obs_model_loss, sepe_obs_model_reconstr_loss = self.f_train_sepe_obs_model(next_observations, observations, actions, eps)
        _, sepe_reward_model_loss, sepe_reward_model_reconstr_loss = self.f_train_sepe_reward_model(rewards, observations, actions, eps)

        if obs_model_loss > sepe_obs_model_loss:
            self.sepe_obs_model_win += 1
            #print("sepe_obs_model_loss:{}\tobs_model_loss:{}\tself.sepe_obs_model_win:{}\n".format(sepe_obs_model_loss, obs_model_loss, self.sepe_obs_model_win))
            if self.sepe_obs_model_win >= 200:
                _ = self.f_copy_sepe_obs_model()
                print("Copying obs model network...") 
                self.sepe_obs_model_win = 0
                #self.use_jole_obs = 0
        else:
            self.sepe_obs_model_win = 0

        if reward_model_loss > sepe_reward_model_loss:
            self.sepe_reward_model_win += 1
            #print("sepe_reward_model_loss:{}\treward_model_loss:{}\tself.sepe_reward_model_win:{}\n".format(sepe_reward_model_loss, reward_model_loss, self.sepe_reward_model_win))
            if self.sepe_reward_model_win >= 200:
                _ = self.f_copy_sepe_reward_model()
                print("Copying reward model network...")  
                self.sepe_reward_model_win = 0
                #self.use_jole_reward = 0
        else:
            self.sepe_reward_model_win = 0

        """
        #test_model

        for i in range(0):
            observation = observations[i:i+1]
            nobs = next_observations[i:i+1]
            nreward = rewards[i:i+1]
            action = actions[i:i+1]
            delta, next_state = self.f_obs_model_predict(observation, action)
            reward = self.f_reward_model_predict(observation, action)
            real_delta = nobs-observation
            print("state:", observation, "predicted delta:", delta, "real delta:", real_delta)
            print("loss:", np.sum((real_delta-delta)**2))
            print("predicted next state:", next_state, "real next state:", nobs)
            print("predicted reward:", reward, "real next reward:", nreward)
        """

        return obs_model_loss, reward_model_loss, sepe_obs_model_loss, sepe_reward_model_loss, obs_model_reconstr_loss, reward_model_reconstr_loss, sepe_obs_model_reconstr_loss,sepe_reward_model_reconstr_loss, obs_recontr_mean, obs_reconstr_log_sigma, obs_reconstr_log_sigma, obs_variance

    def get_jole_obs_actions(self, itr, observations, actions, obs_upper, obs_lower, action_upper, action_lower, train_jole=True, jole_obs_action_type="observed"):
        
        jole_observations = []
        jole_actions = []
        regu_zs = []
        regu_zs_o = []

        if jole_obs_action_type == "observed":
            jole_observations = observations
            jole_actions = actions
        elif jole_obs_action_type == "random_sample":
            for i in range(self.buffer_batch_size):
                #print(actions)
                jole_observations.append(np.random.uniform(obs_lower, obs_upper))
                jole_actions.append(np.random.uniform(action_upper, action_lower))
        elif jole_obs_action_type == "policy_sample":
            for _ in range(5):
                jole_observations.extend(observations) 
            jole_actions, _ = self.es.get_actions(
                    itr, jole_observations, self.policy)
            regu_zs = np.random.normal(0,1,(self.num_z, 5*self.buffer_batch_size, self.dim_z))
            regu_zs_o = np.random.normal(0,1,(self.num_z, self.buffer_batch_size, self.dim_z))
        else:
            jole_observations = observations
            jole_actions = actions

        jole_loss, jole_qval, jole_ys, jole_ys_before_clip = self.f_cal_jole_loss(jole_observations, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, 1.0, regu_zs)
        jole_loss_observed, _, _, _= self.f_cal_jole_loss(observations, actions, self.jole_clip_return_min, self.jole_clip_return_max, 1.0, regu_zs_o)

        return jole_loss, jole_loss_observed, jole_qval, jole_ys, jole_ys_before_clip, jole_observations, jole_actions, regu_zs

    def sample_transitions(self, itr, samples_data):
        """
        Args:
            itr (int): Iterations.
            samples_data (list): Processed batch data.
        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        return transitions, observations, rewards, actions, next_observations, terminals

    @overrides
    def optimize_policy(self, transitions, observations, rewards, actions, next_observations, terminals, jole_obs, jole_actions, regu_zs):
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

        _, qval_loss, qval = self.f_train_qf(ys, inputs, actions, jole_obs, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, self.use_jole_qf, regu_zs)
        _, action_loss = self.f_train_policy(inputs)

        self.f_update_target()

        return qval_loss, ys, qval, action_loss

    @overrides
    def get_itr_snapshot(self, itr):
        """Return data saved in the snapshot for this iteration."""
        return dict(itr=itr, policy=self.policy)
