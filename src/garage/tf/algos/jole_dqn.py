from dowel import tabular
import numpy as np
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.misc.tensor_utils import normalize_pixel_batch
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class JoleDQN(OffPolicyRLAlgorithm):
    """DQN from https://arxiv.org/pdf/1312.5602.pdf.

    Known as Deep Q-Network, it estimates the Q-value function by deep neural
    networks. It enables Q-Learning to be applied on high complexity
    environments. To deal with pixel environments, numbers of tricks are
    usually needed, e.g. skipping frames and stacking frames as single
    observation.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        qf (object): The q value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        exploration_strategy (garage.np.exploration_strategies.
            ExplorationStrategy): Exploration strategy.
        n_epoch_cycles (int): Epoch cycles.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        buffer_batch_size (int): Batch size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        n_train_steps (int): Training steps.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        qf_lr (float): Learning rate for Q-Function.
        qf_optimizer (tf.Optimizer): Optimizer for Q-Function.
        discount (float): Discount factor for rewards.
        target_network_update_freq (int): Frequency of updating target
            network.
        grad_norm_clipping (float): Maximum clipping value for clipping
            tensor values to a maximum L2-norm. It must be larger than 0.
            If None, no gradient clipping is done. For detail, see
            docstring for tf.clip_by_norm.
        double_q (bool): Bool for using double q-network.
        reward_scale (float): Reward scale.
        input_include_goal (bool): Whether input includes goal.
        smooth_return (bool): Whether to smooth the return.
        name (str): Name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 obs_model,
                 reward_model,
                 terminal_model,
                 replay_buffer,
                 env_name="",
                 exploration_strategy=None,
                 n_epoch_cycles=20,
                 min_buffer_size=int(1e4),
                 buffer_batch_size=64,
                 rollout_batch_size=1,
                 n_train_steps=50,
                 max_path_length=None,
                 qf_lr=0.001,
                 qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                 obs_model_lr = 1e-3,
                 obs_model_optimizer=tf.compat.v1.train.AdamOptimizer,
                 reward_model_lr = 1e-3,
                 reward_model_optimizer=tf.compat.v1.train.AdamOptimizer,
                 jole_optimizer=tf.compat.v1.train.AdamOptimizer,
                 terminal_model_lr = 1e-3,
                 terminal_model_optimizer=tf.compat.v1.train.AdamOptimizer,
                 discount=1.0,
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 double_q=False,
                 reward_scale=1.,
                 input_include_goal=False,
                 smooth_return=False,
                 name='JoleDQN'):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.obs_model_lr = obs_model_lr
        self.obs_model_optimizer = obs_model_optimizer
        self.reward_model_lr = reward_model_lr
        self.reward_model_optimizer = reward_model_optimizer
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.double_q = double_q
        self.reward_model = reward_model
        self.obs_model = obs_model
        self.terminal_model = terminal_model

        self.obs_model_optimizer = obs_model_optimizer
        self.reward_model_optimizer = reward_model_optimizer
        self.jole_optimizer = jole_optimizer
        self.terminal_model_optimizer = terminal_model_optimizer

        self.obs_model_lr = obs_model_lr
        self.reward_model_lr = reward_model_lr
        self.terminal_model_lr = terminal_model_lr

        # clone a target q-function
        self.target_qf = qf.clone('target_qf')
        self.sepe_obs_model = obs_model.clone('sepe_obs_model')
        self.sepe_reward_model = reward_model.clone('sepe_reward_model')

        self.obs_model_loss = []
        self.reward_model_loss = []
        self.sepe_obs_model_loss = []
        self.sepe_reward_model_loss = []
        self.jole_loss = []
        self.epoch_ys = []
        self.epoch_qs = []
        self.epoch_jole_ys = []
        self.epoch_jole_qval = []
        self.terminal_model_loss = []
        self.jole_loss_observed = []
        self.epoch_jole_ys_before_clip = []

        self.sepe_reward_model_win = 0
        self.sepe_obs_model_win = 0
        self.use_jole = 0.0
        self.jole_clip_buffer = 10.0
        self.jole_clip_return_min = -np.inf
        self.jole_clip_return_max = np.inf
        self.env_name = env_name


        super(JoleDQN, self).__init__(
            env_spec=env_spec,
            policy=policy,
            qf=qf,
            exploration_strategy=exploration_strategy,
            min_buffer_size=min_buffer_size,
            n_train_steps=n_train_steps,
            n_epoch_cycles=n_epoch_cycles,
            buffer_batch_size=buffer_batch_size,
            rollout_batch_size=rollout_batch_size,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            discount=discount,
            reward_scale=reward_scale,
            input_include_goal=input_include_goal,
            smooth_return=smooth_return)

    @overrides
    def init_opt(self):
        """
        Initialize the networks and Ops.

        Assume discrete space for dqn, so action dimension
        will always be action_space.n
        """
        action_dim = self.env_spec.action_space.n
        obs_dim = self.env_spec.observation_space.flat_dim

        self.episode_rewards = []
        self.episode_qf_losses = []

        with tf.name_scope(self.name, "input"):
            action_t_ph = tf.compat.v1.placeholder(
                tf.int32, None, name='action')
            reward_t_ph = tf.compat.v1.placeholder(
                tf.float32, None, name='reward')
            done_t_ph = tf.compat.v1.placeholder(tf.float32, None, name='done')
            action = tf.one_hot(action_t_ph, action_dim)
            next_obs = tf.compat.v1.placeholder(
                tf.float32, (None, obs_dim), name='next_observations')

            jole_obs = tf.compat.v1.placeholder(
                tf.float32, (None, obs_dim), name='jole_input_observations')
            jole_actions_discrete = tf.compat.v1.placeholder(
                    tf.int32, None, name='jole_input_action')
            jole_actions = tf.one_hot(jole_actions_discrete, action_dim)
            jole_clip_return_min = tf.compat.v1.placeholder(
                    tf.float32, shape=(), name="jole_clip_return_min")
            jole_clip_return_max = tf.compat.v1.placeholder(
                tf.float32, shape=(), name="jole_clip_return_max")
            use_jole = tf.compat.v1.placeholder(
                tf.float32, shape=(), name="use_jole")
            obs = self.qf.input

        # set up jole
        with tf.name_scope(self.name, "jole"):
            #get Q(s,a)
            jole_qval = tf.reduce_sum(self.qf.get_qval_sym(jole_obs, name='jole_q_value') * jole_actions, axis=1)
            # get predicted next observations and actions
            jole_predicted_next_obs = tf.reshape(tf.reduce_sum(tf.reshape(self.obs_model.get_fval_sym(jole_obs, name='jole_obs_value'),
                                            shape=(-1, action_dim, obs_dim)) * tf.expand_dims(jole_actions,-1), axis=1),shape=(-1, obs_dim))
            jole_predicted_reward = tf.reduce_sum(self.reward_model.get_fval_sym(jole_obs, name='jole_reward_value')*jole_actions, axis=1)
            jole_predicted_terminal = self.get_terminal_status(jole_predicted_next_obs)
            
            #jole_predicted_terminal = 0
            #jole_predicted_terminal = tf.argmax(self.terminal_model.get_fval_sym(jole_predicted_next_obs, name='jole_terminal_value'), axis=-1)

            # r + Q'(s', argmax_a(Q(s', _)) - Q(s, a)
            if self.double_q:
                jole_target_qval_with_online_q = self.qf.get_qval_sym(
                    jole_predicted_next_obs, name="jole_next_obs_value")
                jole_future_best_q_val_action = tf.argmax(
                    jole_target_qval_with_online_q, 1)
                jole_future_best_q_val = tf.reduce_sum(
                    self.target_qf.get_qval_sym(jole_predicted_next_obs, name="jole_next_obs_value") * tf.one_hot(
                        jole_future_best_q_val_action, action_dim),
                    axis=1)
            else:
                # r + max_a(Q'(s', _)) - Q(s, a)
                jole_future_best_q_val = tf.reduce_max(
                    self.target_qf.get_qval_sym(jole_predicted_next_obs, name="jole_next_obs_value"), axis=1)
            #jole_done_t_ph = tf.condition
            jole_q_best_masked = (1.0 - tf.cast(jole_predicted_terminal, tf.float32)) * jole_future_best_q_val
            #jole_q_best_masked = jole_future_best_q_val
            # if done, it's just reward
            # else reward + discount * future_best_q_val
            jole_target_q_values_before_clip = (jole_predicted_reward + self.discount * jole_q_best_masked)
            jole_target_q_values = jole_target_q_values_before_clip#tf.clip_by_value(jole_target_q_values_before_clip, jole_clip_return_min, jole_clip_return_max)

            jole_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(jole_qval, jole_target_q_values))
            
            self.f_cal_jole_loss = tensor_utils.compile_function(
                inputs=[jole_obs, jole_actions_discrete, jole_clip_return_min, jole_clip_return_max, use_jole],
                outputs=[jole_loss, jole_qval, jole_target_q_values, jole_target_q_values_before_clip])

        #train the env model
        with tf.name_scope(self.name, "env_model"):

            predicted_next_obs = tf.reduce_sum(tf.reshape(self.obs_model.get_fval_sym(obs, name='obs_value'),
                                            shape=(-1, action_dim, obs_dim)) * tf.expand_dims(action, -1), axis=1)
            predicted_reward = tf.reduce_sum(self.reward_model.get_fval_sym(obs, name='reward_value')*action, axis=1)

            #change to predict the delta of s
            original_obs_model_loss = tf.reduce_mean(
                tf.compat.v1.squared_difference(next_obs, predicted_next_obs))
            obs_model_loss = original_obs_model_loss + use_jole * 0.0001 * jole_loss
            original_reward_model_loss = tf.reduce_mean(
                tf.compat.v1.squared_difference(reward_t_ph, predicted_reward))
            reward_model_loss = original_reward_model_loss + use_jole * 0.0001 * jole_loss

            predicted_terminal = self.terminal_model.get_fval_sym(next_obs, name="terminal_value")
            terminal_model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_terminal, labels=tf.cast(tf.squeeze(done_t_ph), dtype=tf.int32))

            terminal_model_accurate = tf.reduce_sum(1 - tf.abs(tf.argmax(predicted_terminal, axis=-1) - tf.cast(tf.squeeze(done_t_ph), dtype=tf.int64)))

            with tf.name_scope('minimize_obs_model_loss'):
                obs_train_op = self.obs_model_optimizer(
                    self.obs_model_lr, name='ObsModelOptimizer').minimize(
                    obs_model_loss, var_list=self.obs_model.get_trainable_vars())
                reward_train_op = self.reward_model_optimizer(
                    self.reward_model_lr, name='RewardModelOptimizer').minimize(
                    reward_model_loss, var_list=self.reward_model.get_trainable_vars())
                terminal_train_op = self.terminal_model_optimizer(
                    self.terminal_model_lr, name='TerminalModelOptimizer').minimize(
                    terminal_model_loss, var_list=self.terminal_model.get_trainable_vars())

            self.f_train_obs_model = tensor_utils.compile_function(
                inputs=[next_obs, obs, action_t_ph, jole_obs, jole_actions_discrete, jole_clip_return_min, jole_clip_return_max, use_jole],
                outputs=[obs_train_op, obs_model_loss, original_obs_model_loss])
            self.f_train_reward_model = tensor_utils.compile_function(
                inputs=[reward_t_ph, obs, action_t_ph, jole_obs, jole_actions_discrete, jole_clip_return_min, jole_clip_return_max, use_jole],
                outputs=[reward_train_op, reward_model_loss, original_reward_model_loss])
            self.f_train_terminal_model = tensor_utils.compile_function(
                inputs=[next_obs, done_t_ph],
                outputs=[terminal_train_op, terminal_model_loss, terminal_model_accurate])
            self.f_obs_model_predict = tensor_utils.compile_function(
                inputs=[obs, action_t_ph],
                outputs=[predicted_next_obs-obs, predicted_next_obs])
            self.f_reward_model_predict = tensor_utils.compile_function(
                inputs=[obs, action_t_ph],
                outputs=[predicted_reward])
            self.f_terminal_model_predict = tensor_utils.compile_function(
                inputs=[next_obs],
                outputs=[predicted_terminal, tf.argmax(predicted_terminal, axis=-1)])

            sepe_predicted_next_obs = tf.reduce_sum(tf.reshape(self.sepe_obs_model.get_fval_sym(obs, name='obs_value'),
                                            shape=(-1, action_dim, obs_dim)) * tf.expand_dims(action, -1), axis=1)
            sepe_predicted_reward = tf.reduce_sum(self.sepe_reward_model.get_fval_sym(obs, name='reward_value')*action, axis=1)
            #change to predict the delta of s
            sepe_obs_model_loss = tf.reduce_mean(
                tf.compat.v1.squared_difference(next_obs, sepe_predicted_next_obs))
            sepe_reward_model_loss = tf.reduce_mean(
                tf.compat.v1.squared_difference(reward_t_ph, sepe_predicted_reward))

            with tf.name_scope('minimize_sepe_obs_model_loss'):
                sepe_obs_train_op = self.obs_model_optimizer(
                    self.obs_model_lr, name='SepeObsModelOptimizer').minimize(
                    sepe_obs_model_loss, var_list=self.sepe_obs_model.get_trainable_vars())
                sepe_reward_train_op = self.reward_model_optimizer(
                    self.reward_model_lr, name='SepeRewardModelOptimizer').minimize(
                    sepe_reward_model_loss, var_list=self.sepe_reward_model.get_trainable_vars())

            f_train_sepe_obs_model = tensor_utils.compile_function(
                    inputs=[next_obs, obs, action_t_ph],
                    outputs=[sepe_obs_train_op, sepe_obs_model_loss])
            f_train_sepe_reward_model = tensor_utils.compile_function(
                inputs=[reward_t_ph, obs, action_t_ph],
                outputs=[sepe_reward_train_op, sepe_reward_model_loss])

            self.f_train_sepe_obs_model = f_train_sepe_obs_model
            self.f_train_sepe_reward_model = f_train_sepe_reward_model

            # Copy the parameter of seperate env models when necessary
            with tf.name_scope('copy_sepe_env_models'):
                copy_sepe_obs_model_ops = tensor_utils.get_target_ops(
                    self.sepe_obs_model.get_global_vars(),
                    self.obs_model.get_global_vars())

                copy_sepe_reward_model_ops = tensor_utils.get_target_ops(
                    self.sepe_reward_model.get_global_vars(),
                    self.reward_model.get_global_vars())

            self.f_copy_sepe_obs_model = tensor_utils.compile_function(
                inputs=[], outputs=copy_sepe_obs_model_ops)
            self.f_copy_sepe_reward_model = tensor_utils.compile_function(
                inputs=[], outputs=copy_sepe_reward_model_ops)

        # build q networks
        with tf.name_scope(self.name, 'DQN'):
            with tf.name_scope('update_ops'):
                target_update_op = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self.target_qf.get_global_vars())

            self._qf_update_ops = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('td_error'):
                # Q-value of the selected action
                q_selected = tf.reduce_sum(
                    self.qf.q_vals * action,  # yapf: disable
                    axis=1)

                # r + Q'(s', argmax_a(Q(s', _)) - Q(s, a)
                if self.double_q:
                    target_qval_with_online_q = self.qf.get_qval_sym(
                        self.target_qf.input, self.qf.name)
                    future_best_q_val_action = tf.argmax(
                        target_qval_with_online_q, 1)
                    future_best_q_val = tf.reduce_sum(
                        self.target_qf.q_vals * tf.one_hot(
                            future_best_q_val_action, action_dim),
                        axis=1)
                else:
                    # r + max_a(Q'(s', _)) - Q(s, a)
                    future_best_q_val = tf.reduce_max(
                        self.target_qf.q_vals, axis=1)

                q_best_masked = (1.0 - done_t_ph) * future_best_q_val
                # if done, it's just reward
                # else reward + discount * future_best_q_val
                target_q_values = (reward_t_ph + self.discount * q_best_masked)

                # td_error = q_selected - tf.stop_gradient(target_q_values)
                loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(tf.stop_gradient(target_q_values), q_selected))
                #loss = tf.compat.v1.losses.huber_loss(
                #    q_selected, tf.stop_gradient(target_q_values))
                #loss = tf.reduce_mean(loss)
                loss += use_jole * 0.2 * jole_loss

            with tf.name_scope('optimize_ops'):
                optimizer = self.qf_optimizer(self.qf_lr)
                if self.grad_norm_clipping is not None:
                    gradients = optimizer.compute_gradients(
                        loss, var_list=self.qf.get_trainable_vars())
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            gradients[i] = (tf.clip_by_norm(
                                grad, self.grad_norm_clipping), var)
                        optimize_loss = optimizer.apply_gradients(gradients)
                else:
                    optimize_loss = optimizer.minimize(
                        loss, var_list=self.qf.get_trainable_vars())

            self._train_qf = tensor_utils.compile_function(
                inputs=[
                    self.qf.input, action_t_ph, reward_t_ph, done_t_ph,
                    self.target_qf.input, jole_obs, jole_actions_discrete, use_jole, jole_clip_return_max, jole_clip_return_min
                ],
                outputs=[loss, optimize_loss, q_selected, target_q_values])

            for variable in tf.trainable_variables():
                print(variable)

    def get_terminal_status(self, jole_predicted_next_obs):
        jole_predicted_terminal = tf.zeros_like(jole_predicted_next_obs)
        if self.env_name == "MountainCar-v0":
            jole_predicted_terminal = (jole_predicted_next_obs[:, 0]>0.5)
        return jole_predicted_terminal


    def train_once(self, itr, paths, obs_upper, obs_lower, action_upper, action_lower, evaluate_paths):
        """Train the algorithm once."""
        paths = self.process_samples(itr, paths)
        evaluate_paths = self.process_samples(itr, evaluate_paths)
        epoch = itr / self.n_epoch_cycles
        if itr>600:
            self.use_jole_qf = 1.0
            self.use_jole_obs = 1.0
            self.use_jole_reward = 1.0
        else:
            self.use_jole_qf = 0.0
            self.use_jole_obs = 0.0
            self.use_jole_reward = 0.0

        self.episode_rewards.extend(evaluate_paths['undiscounted_returns'])
        last_average_return = np.mean(self.episode_rewards)
        self.jole_clip_return_min = np.min(self.episode_rewards) - self.jole_clip_buffer
        self.jole_clip_return_max = np.max(self.episode_rewards) + self.jole_clip_buffer
        for train_itr in range(self.n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self.min_buffer_size):
                self.evaluate = True
                transitions, observations, rewards, actions, next_observations, terminals = self.sample_transitions(itr, paths)
                jole_loss, jole_loss_observed, jole_qval, jole_ys, jole_ys_before_clip, jole_obs, jole_actions = self.get_jole_obs_actions(itr, observations, actions, train_jole = True, jole_obs_action_type = "policy_sample")
                qf_loss, q, y = self.optimize_policy(itr, observations, rewards, actions, next_observations, terminals, jole_obs, jole_actions)
                obs_model_loss, reward_model_loss, sepe_obs_model_loss, sepe_reward_model_loss, terminal_model_loss= self.optimize_model(itr, jole_obs, jole_actions, train_itr)

                self.epoch_ys.append(y)
                self.epoch_qs.append(q)
                self.epoch_jole_ys_before_clip.append(jole_ys_before_clip)
                self.epoch_jole_qval.append(jole_qval)
                self.episode_qf_losses.append(qf_loss)
                self.epoch_jole_ys.append(jole_ys)
                self.epoch_jole_qval.append(jole_qval)
                self.reward_model_loss.append(reward_model_loss)
                self.obs_model_loss.append(obs_model_loss)
                self.sepe_reward_model_loss.append(sepe_reward_model_loss)
                self.sepe_obs_model_loss.append(sepe_obs_model_loss)
                self.jole_loss.append(jole_loss)
                self.jole_loss_observed.append(jole_loss_observed)
                self.terminal_model_loss.append(terminal_model_loss)

        if self.evaluate:
            if itr % self.target_network_update_freq == 0:
                self._qf_update_ops()

        if itr % self.n_epoch_cycles == 0:
            if self.evaluate:
                print(itr)
                print(self.use_jole_qf)
                mean100ep_rewards = round(
                    np.mean(self.episode_rewards[-100:]), 1)
                mean100ep_qf_loss = np.mean(self.episode_qf_losses[-100:])
                tabular.record('Epoch', epoch)
                tabular.record('AverageReturn', np.mean(self.episode_rewards))
                tabular.record('StdReturn', np.std(self.episode_rewards))
                tabular.record('Episode100RewardMean', mean100ep_rewards)
                tabular.record('{}/Episode100LossMean'.format(self.qf.name),
                               mean100ep_qf_loss)
                tabular.record('QFunction/AverageQ', np.mean(self.epoch_qs))
                tabular.record('QFunction/MaxQ', np.max(self.epoch_qs))
                tabular.record('QFunction/AverageAbsQ',
                               np.mean(np.abs(self.epoch_qs)))
                tabular.record('QFunction/AverageY', np.mean(self.epoch_ys))
                tabular.record('QFunction/MaxY', np.max(self.epoch_ys))
                tabular.record('QFunction/AverageAbsY',
                               np.mean(np.abs(self.epoch_ys)))
                tabular.record('QFunction/AverageJoleQ', np.mean(self.epoch_jole_qval))
                tabular.record('QFunction/MaxJoleQ', np.max(self.epoch_jole_qval))
                tabular.record('QFunction/AverageAbsJoleQ',
                               np.mean(np.abs(self.epoch_jole_qval)))
                tabular.record('QFunction/AverageJoleY', np.mean(self.epoch_jole_ys))
                tabular.record('QFunction/MaxJoleY', np.max(self.epoch_jole_ys))
                tabular.record('QFunction/AverageAbsJoleY',
                               np.mean(np.abs(self.epoch_jole_ys)))
                tabular.record('QFunction/AverageJoleYBeforeClip', np.mean(self.epoch_jole_ys_before_clip))
                tabular.record('QFunction/MaxJoleYBeforeClip', np.max(self.epoch_jole_ys_before_clip))
                tabular.record('QFunction/AverageAbsJoleYBeforeClip',
                               np.mean(np.abs(self.epoch_jole_ys_before_clip)))
                tabular.record('Jole/ObsModelLoss',
                               np.mean(self.obs_model_loss))
                tabular.record('Jole/RewardModelLoss',
                               np.mean(self.reward_model_loss))
                tabular.record('Jole/SepeObsModelLoss',
                               np.mean(self.sepe_obs_model_loss))
                tabular.record('Jole/SepeRewardModelLoss',
                               np.mean(self.sepe_reward_model_loss))
                tabular.record('Jole/TerminalModelLoss',
                               np.mean(self.terminal_model_loss))
                tabular.record('Jole/JoleLoss',
                               np.mean(self.jole_loss))
                tabular.record('Jole/JoleLossObserved',
                               np.mean(self.jole_loss_observed))

        if not self.smooth_return:
            self.episode_rewards = []
            self.episode_qf_losses = []
            self.epoch_ys = []
            self.epoch_qs = []
            self.epoch_jole_ys = []
            self.epoch_jole_qval = []
            self.obs_model_loss = []
            self.reward_model_loss = []
            self.sepe_obs_model_loss = []
            self.sepe_reward_model_loss = []
            self.jole_loss = []
            self.jole_loss_observed = []
            self.epoch_jole_ys_before_clip = []
            self.terminal_model_loss = []

        #self.success_history.clear()
        return last_average_return

    @overrides
    def get_itr_snapshot(self, itr):
        """Get snapshot of the policy."""
        return dict(itr=itr, policy=self.policy)

    @overrides
    def optimize_policy(self, itr, observations, rewards, actions, next_observations, dones, jole_obs, jole_actions):
        """Optimize network using experiences from replay buffer."""
        # normalize pixel to range [0, 1] since the samples stored in the
        # replay buffer are of type uint8 and not normalized, for memory
        # optimization
        observations = normalize_pixel_batch(self.env_spec, observations)
        next_observations = normalize_pixel_batch(self.env_spec,
                                                  next_observations)
        loss, _ , qval, y= self._train_qf(observations, actions, rewards, dones,
                                 next_observations, jole_obs, jole_actions, self.use_jole_qf, self.jole_clip_return_max, self.jole_clip_return_min)
        return loss, qval, y

    def __getstate__(self):
        data = self.__dict__.copy()
        del data['_qf_update_ops']
        del data['_train_qf']
        return data

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_opt()

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

        #rewards = rewards.reshape(-1, 1)
        #terminals = terminals.reshape(-1, 1)

        return transitions, observations, rewards, actions, next_observations, terminals

    def get_jole_obs_actions(self, itr, observations, actions, train_jole=True, jole_obs_action_type="observed"):
        
        jole_observations = []
        jole_actions = []

        if jole_obs_action_type == "observed":
            jole_observations = observations
            jole_actions = actions
        elif jole_obs_action_type == "policy_sample":
            jole_observations = observations
            jole_actions, _ = self.es.get_actions_no_decay(
                    itr, jole_observations, self.policy)
        else:
            jole_observations = observations
            jole_actions = actions

        jole_loss, jole_qval, jole_ys, jole_ys_before_clip = self.f_cal_jole_loss(jole_observations, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, 1.0)
        jole_loss_observed, _, _, _ = self.f_cal_jole_loss(observations, actions, self.jole_clip_return_min, self.jole_clip_return_max, 1.0)
        return jole_loss, jole_loss_observed, jole_qval, jole_ys, jole_ys_before_clip, jole_observations, jole_actions

    def optimize_model(self, itr, jole_obs, jole_actions, times=1):
        """Perform model optimizing
        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        dones = transitions['terminal']

        _, _, obs_model_loss = self.f_train_obs_model(next_observations, observations, actions, jole_obs, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, self.use_jole_obs)
        _, _, reward_model_loss = self.f_train_reward_model(rewards, observations, actions, jole_obs, jole_actions, self.jole_clip_return_min, self.jole_clip_return_max, self.use_jole_reward)

        _, sepe_obs_model_loss = self.f_train_sepe_obs_model(next_observations, observations, actions)
        _, sepe_reward_model_loss = self.f_train_sepe_reward_model(rewards, observations, actions)

        _, terminal_model_loss, terminal_model_accurate = self.f_train_terminal_model(next_observations, dones)
        if itr %20 ==0 and times % 100 == 0:
            #predicted_terminal, predicted_terminal_label = self.f_terminal_model_predict(next_observations)
            #print(dones, predicted_terminal_label)
            #print("total length:", len(next_observations), "accurate:", terminal_model_accurate)
            obs_delta, predicted_next_obs = self.f_obs_model_predict([observations[0]], [actions[0]])
            predicted_reward = self.f_reward_model_predict([observations[0]], [actions[0]])
            print("obs and action:", observations[0], actions[0])
            print("ground truth delta_obs, next_obs, reward", next_observations[0]-observations[0], next_observations[0], rewards[0])
            print("predicted delta_obs, next_obs, reward", obs_delta, predicted_next_obs, predicted_reward)

        if obs_model_loss > sepe_obs_model_loss:
            self.sepe_obs_model_win += 1
            #print("sepe_obs_model_loss:{}\tobs_model_loss:{}\tself.sepe_obs_model_win:{}\n".format(sepe_obs_model_loss, obs_model_loss, self.sepe_obs_model_win))
            if self.sepe_obs_model_win >= 200:
                #_ = self.f_copy_sepe_obs_model()
                print("Copying obs model network...") 
                self.sepe_obs_model_win = 0
                #self.use_jole_obs = 0
        else:
            self.sepe_obs_model_win = 0

        if reward_model_loss > sepe_reward_model_loss:
            self.sepe_reward_model_win += 1
            #print("sepe_reward_model_loss:{}\treward_model_loss:{}\tself.sepe_reward_model_win:{}\n".format(sepe_reward_model_loss, reward_model_loss, self.sepe_reward_model_win))
            if self.sepe_reward_model_win >= 200:
                #_ = self.f_copy_sepe_reward_model()
                print("Copying reward model network...")  
                self.sepe_reward_model_win = 0
                #self.use_jole_reward = 0
        else:
            self.sepe_reward_model_win = 0
        self.policy.__update__(self.qf)

        return obs_model_loss, reward_model_loss, sepe_obs_model_loss, sepe_reward_model_loss, terminal_model_loss
