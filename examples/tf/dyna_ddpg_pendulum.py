#!/usr/bin/env python3
"""
This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import gym
import tensorflow as tf

from garage.experiment import run_experiment
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos.dyna_ddpg import DynaDDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.tf.env_functions.continuous_mlp_obs_function import ContinuousMLPObsFunction
from garage.tf.env_functions.continuous_mlp_reward_function import ContinuousMLPRewardFunction

env_name = 'HalfCheetah-v3'
def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('HalfCheetah-v3'))

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu)

        reward_model = ContinuousMLPRewardFunction(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu)

        obs_model = ContinuousMLPObsFunction(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)

        ima_replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)

        dyna_ddpg = DynaDDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    reward_model = reward_model,
                    obs_model = obs_model,
                    replay_buffer=replay_buffer,
                    ima_replay_buffer=ima_replay_buffer,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.99,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.train.AdamOptimizer,
                    qf_optimizer=tf.train.AdamOptimizer)

        runner.setup(algo=dyna_ddpg, env=env)

        runner.train(n_epochs=500, n_epoch_cycles=20, batch_size=100)


for i in range(1,2):
    run_experiment(
        run_task,
        snapshot_mode='none',
        seed=i,
        log_dir='data/type_ddpg/{}/dyna_ddpg/{}'.format(env_name, i)
    )
