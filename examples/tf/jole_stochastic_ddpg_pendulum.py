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

from garage.envs.wrappers.reverse_action import ReverseAction
from garage.envs.wrappers.double_action import DoubleAction
from garage.experiment import run_experiment
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos.jole_ddpg_stochastic import JoLeDDPGStochastic
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.tf.env_functions.cvae_obs_generator_reward import CVAERewardGenerator
from garage.tf.env_functions.cvae_obs_recognition_reward import CVAERewardRecognition
from garage.tf.env_functions.cvae_obs_generator import CVAEObsGenerator
from garage.tf.env_functions.cvae_obs_recognition import CVAEObsRecognition

def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('HalfCheetah-v3'))
        env = ReverseAction(env)

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu)

        reward_model_generator = CVAERewardGenerator(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu,
                                    z_dim=1)

        reward_model_recognition = CVAERewardRecognition(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu,
                                    z_dim=1)

        obs_model_generator = CVAEObsGenerator(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu,
                                    z_dim=1)

        obs_model_recognition = CVAEObsRecognition(env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    hidden_nonlinearity=tf.nn.relu,
                                    z_dim=1)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)

        jole_ddpg = JoLeDDPGStochastic(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    reward_model_generator = reward_model_generator,
                    reward_model_recognition = reward_model_recognition,
                    obs_model_generator = obs_model_generator,
                    obs_model_recognition = obs_model_recognition,
                    replay_buffer=replay_buffer,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.99,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.train.AdamOptimizer,
                    qf_optimizer=tf.train.AdamOptimizer,
                    z_dim=1)

        runner.setup(algo=jole_ddpg, env=env)

        runner.train(n_epochs=500, n_epoch_cycles=20, batch_size=100)
env_name = "HalfCheetah-v3"

for i in range(1, 6):
    run_experiment(
        run_task,
        snapshot_mode='none',
        seed=i,
        log_dir="data/type_stochastic_ddpg/{}/jole_ddpg_with_sigma/reverse_action/{}_local".format(env_name,i)
    )
