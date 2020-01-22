#!/usr/bin/env python3
"""
An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
import gym
import tensorflow as tf

from garage.experiment import run_experiment
from garage.np.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos.jole_dqn import JoleDQN
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from garage.tf.env_functions.discrete_mlp_obs_function import DiscreteMLPObsFunction
from garage.tf.env_functions.discrete_mlp_reward_function import DiscreteMLPRewardFunction
from garage.tf.env_functions.mlp_terminal_function import MLPTerminalFunction

env_name = 'MountainCar-v0'
def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        n_epochs = 500
        n_epoch_cycles = 20
        sampler_batch_size = 100
        num_timesteps = n_epochs * n_epoch_cycles * sampler_batch_size
        env_name = 'MountainCar-v0'
        env = TfEnv(gym.make(env_name))
        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e4),
                                           time_horizon=1)

        qf = DiscreteMLPQFunction(env_spec=env.spec, 
                                  hidden_sizes=(20,),
                                  hidden_nonlinearity=tf.nn.relu)

        obs_model = DiscreteMLPObsFunction(env_spec=env.spec, 
                                           hidden_sizes=(20,),
                                           hidden_nonlinearity=tf.nn.relu)

        reward_model = DiscreteMLPRewardFunction(env_spec=env.spec, 
                                                hidden_sizes=(20,),
                                                hidden_nonlinearity=tf.nn.relu)
        #terminal model for predicting the end of an episode
        terminal_model = MLPTerminalFunction(env_spec=env.spec, 
                                            hidden_sizes=(20,),
                                            hidden_nonlinearity=tf.nn.relu)

        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)

        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_timesteps=num_timesteps,
            max_epsilon=0.5,
            min_epsilon=0.01,
            decay_ratio=0.1)

        algo = JoleDQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   obs_model = obs_model,
                   reward_model = reward_model,
                   terminal_model = terminal_model,
                   exploration_strategy=epilson_greedy_strategy,
                   replay_buffer=replay_buffer,
                   qf_lr=1e-3,
                   discount=0.99,
                   min_buffer_size=int(1e3),
                   double_q=False,
                   n_train_steps=50,
                   n_epoch_cycles=n_epoch_cycles,
                   target_network_update_freq=100,
                   buffer_batch_size=64,
                   env_name=env_name)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs,
                     n_epoch_cycles=n_epoch_cycles,
                     batch_size=sampler_batch_size)
for i in range(1,2):
  run_experiment(run_task, 
                snapshot_mode='none', 
                seed=i, 
                log_dir="data/type_dqn/{}/jole_dqn/{}/".format(env_name,i))
