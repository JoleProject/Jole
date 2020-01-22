"""
This module implements a Vectorized Sampler used for OffPolicy Algorithms.

It diffs from OnPolicyVectorizedSampler in two parts:
 - The num of envs is defined by rollout_batch_size. In
 OnPolicyVectorizedSampler, the number of envs can be decided by batch_size
 and max_path_length. But OffPolicy algorithms usually samples transitions
 from replay buffer, which only has buffer_batch_size.
 - It needs to add transitions to replay buffer throughout the rollout.
"""
import itertools
import pickle

import numpy as np

from garage.experiment import deterministic
from garage.misc import tensor_utils
from garage.misc.overrides import overrides
from garage.sampler.batch_sampler import BatchSampler
from garage.sampler.vec_env_executor import VecEnvExecutor


class OffPolicyVectorizedSampler(BatchSampler):
    """This class implements OffPolicyVectorizedSampler.

    Args:
        algo (garage.np.RLAlgorithm): Algorithm.
        env (garage.envs.GarageEnv): Environment.
        n_envs (int): Number of parallel environments managed by sampler.
        no_reset (bool): Reset environment between samples or not.

    """

    def __init__(self, algo, env, n_envs=None, no_reset=True):
        if n_envs is None:
            n_envs = int(algo.rollout_batch_size)
        super().__init__(algo, env)
        self.n_envs = n_envs
        self.no_reset = no_reset

        self._last_obses = None
        self._last_uncounted_discount = [0] * n_envs
        self._last_running_length = [0] * n_envs
        self._last_success_count = [0] * n_envs

        self._obs_upper = None
        self._obs_lower = None
        self._action_upper = None
        self._action_lower = None
        self._bound_start = False

    @overrides
    def start_worker(self):
        """Initialize the sampler."""
        n_envs = self.n_envs
        envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]
        eval_envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]

        # Deterministically set environment seeds based on the global seed.
        for (i, e) in enumerate(envs):
            e.seed(deterministic.get_seed() + i)

        self.vec_env = VecEnvExecutor(
            envs=envs, max_path_length=self.algo.max_path_length)
        self.evaluate_env = VecEnvExecutor(
            envs=eval_envs, max_path_length=self.algo.max_path_length)
        self.env_spec = self.env.spec

    @overrides
    def shutdown_worker(self):
        """Terminate workers if necessary."""
        self.vec_env.close()
        self.evaluate_env.close()

    def obtain_samples_for_evaluation(self, num_paths=20):
        """Collect samples for the given iteration number.

        Args:
            itr(int): Iteration number.
            batch_size(int): Number of environment interactions in one batch.

        Returns:
            list: A list of paths.

        """
        paths = []

        policy = self.algo.policy

        for i in range(num_paths):
            obses = self.evaluate_env.reset()
            #print(obses)

            dones = np.asarray([True] * self.evaluate_env.num_envs)
            running_paths = [None] * self.evaluate_env.num_envs
            policy.reset(dones)
            end_of_path = False

            for j in range(500):
                input_obses = obses
                obs_normalized = tensor_utils.normalize_pixel_batch(
                        self.env_spec, input_obses)
                obses = obs_normalized 
                
                actions = self.algo.policy.get_actions(
                        obs_normalized)
                if len(actions) > 1:
                    actions = actions[0]
                agent_infos = None

                next_obses, rewards, dones, env_infos = self.evaluate_env.step(actions)
                original_next_obses = next_obses
                next_obses = tensor_utils.normalize_pixel_batch(
                    self.env_spec, next_obses)
                
                env_infos = tensor_utils.split_tensor_dict_list(env_infos)

                if agent_infos is None:
                    agent_infos = [dict() for _ in range(self.evaluate_env.num_envs)]
                if env_infos is None:
                    env_infos = [dict() for _ in range(self.evaluate_env.num_envs)]


                for idx, reward, env_info, done in zip(itertools.count(), rewards,
                                                       env_infos, dones):
                    if running_paths[idx] is None:
                        running_paths[idx] = dict(
                            rewards=[],
                            env_infos=[],
                            dones=[],
                            undiscounted_return=0,
                            # running_length: Length of path up to now
                            # Note that running_length is not len(rewards)
                            # Because a path may not be complete in one batch
                            running_length=0,
                            success_count=0)

                    running_paths[idx]['rewards'].append(reward)
                    running_paths[idx]['env_infos'].append(env_info)
                    running_paths[idx]['dones'].append(done)
                    running_paths[idx]['running_length'] += 1
                    running_paths[idx]['undiscounted_return'] += reward
                    running_paths[idx]['success_count'] += env_info.get(
                        'is_success') or 0

                    if done or j == 499:
                        paths.append(
                            dict(
                                rewards=tensor_utils.stack_tensor_list(
                                    running_paths[idx]['rewards']),
                                dones=tensor_utils.stack_tensor_list(
                                    running_paths[idx]['dones']),
                                env_infos=tensor_utils.stack_tensor_dict_list(
                                    running_paths[idx]['env_infos']),
                                running_length=running_paths[idx]
                                ['running_length'],
                                undiscounted_return=running_paths[idx]
                                ['undiscounted_return'],
                                success_count=running_paths[idx]['success_count']))
                        running_paths[idx] = None

                        end_of_path = True
                if end_of_path:
                    break
                obses = original_next_obses
        #print(paths)
        return paths


    @overrides
    def obtain_samples(self, itr, batch_size, is_evaluate=False):
        """Collect samples for the given iteration number.

        Args:
            itr(int): Iteration number.
            batch_size(int): Number of environment interactions in one batch.

        Returns:
            list: A list of paths.

        """
        paths = []
        if not self.no_reset or self._last_obses is None:
            obses = self.vec_env.reset()
        else:
            obses = self._last_obses
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs
        n_samples = 0

        policy = self.algo.policy
        if self.algo.es:
            self.algo.es.reset()

        while n_samples < batch_size:
            policy.reset(dones)
            if self.algo.input_include_goal:
                obs = [obs['observation'] for obs in obses]
                d_g = [obs['desired_goal'] for obs in obses]
                a_g = [obs['achieved_goal'] for obs in obses]
                input_obses = np.concatenate((obs, d_g), axis=-1)
            else:               
                input_obses = obses

            obs_normalized = tensor_utils.normalize_pixel_batch(
                    self.env_spec, input_obses)
            obses = obs_normalized 
            
            if self.algo.es and not is_evaluate:
                actions, agent_infos = self.algo.es.get_actions(
                    itr, obs_normalized, self.algo.policy)
                agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            else:
                actions = self.algo.policy.get_actions(
                    obs_normalized)
                if len(actions) > 1:
                    actions = actions[0]
                agent_infos = None

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            original_next_obses = next_obses
            next_obses = tensor_utils.normalize_pixel_batch(
                self.env_spec, next_obses)

            self._last_obses = next_obses
            
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            n_samples += len(next_obses)

            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]

            if self.algo.input_include_goal:
                self.algo.replay_buffer.add_transitions(
                    observation=obs,
                    action=actions,
                    goal=d_g,
                    achieved_goal=a_g,
                    terminal=dones,
                    next_observation=[
                        next_obs['observation'] for next_obs in next_obses
                    ],
                    next_achieved_goal=[
                        next_obs['achieved_goal'] for next_obs in next_obses
                    ],
                )
            else:
                self.algo.replay_buffer.add_transitions(
                    observation=obs_normalized,
                    action=actions,
                    reward=rewards * self.algo.reward_scale,
                    terminal=dones,
                    next_observation=next_obses,
                )

            if self._bound_start == False:
                self._bound_start = True
                self._obs_upper = obses[0]
                self._obs_lower = obses[0]
                self._action_upper = actions[0]
                self._action_lower = actions[0]

            for obs in obses:
                self._obs_upper = np.maximum(self._obs_upper, obs)
                self._obs_lower = np.minimum(self._obs_lower, obs)
            for action in actions:
                self._action_upper = np.maximum(self._action_upper, action)
                self._action_lower = np.minimum(self._action_lower, action)

            for idx, reward, env_info, done in zip(itertools.count(), rewards,
                                                   env_infos, dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        rewards=[],
                        env_infos=[],
                        dones=[],
                        undiscounted_return=self._last_uncounted_discount[idx],
                        # running_length: Length of path up to now
                        # Note that running_length is not len(rewards)
                        # Because a path may not be complete in one batch
                        running_length=self._last_running_length[idx],
                        success_count=self._last_success_count[idx])

                running_paths[idx]['rewards'].append(reward)
                running_paths[idx]['env_infos'].append(env_info)
                running_paths[idx]['dones'].append(done)
                running_paths[idx]['running_length'] += 1
                running_paths[idx]['undiscounted_return'] += reward
                running_paths[idx]['success_count'] += env_info.get(
                    'is_success') or 0

                self._last_uncounted_discount[idx] += reward
                self._last_success_count[idx] += env_info.get(
                    'is_success') or 0
                self._last_running_length[idx] += 1

                if done or n_samples >= batch_size:
                    paths.append(
                        dict(
                            rewards=tensor_utils.stack_tensor_list(
                                running_paths[idx]['rewards']),
                            dones=tensor_utils.stack_tensor_list(
                                running_paths[idx]['dones']),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[idx]['env_infos']),
                            running_length=running_paths[idx]
                            ['running_length'],
                            undiscounted_return=running_paths[idx]
                            ['undiscounted_return'],
                            success_count=running_paths[idx]['success_count']))
                    running_paths[idx] = None

                    if done:
                        self._last_running_length[idx] = 0
                        self._last_success_count[idx] = 0
                        self._last_uncounted_discount[idx] = 0

                    if self.algo.es:
                        self.algo.es.reset()
            obses = original_next_obses
        return paths, self._obs_upper, self._obs_lower, self._action_upper, self._action_lower