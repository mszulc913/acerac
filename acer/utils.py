from typing import Tuple, List, Union, Dict

import gym
import numpy as np
import tensorflow as tf


def normc_initializer():
    """Normalized column initializer."""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape)
        out *= 1 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out, dtype=dtype)
    return _initializer


def flatten_experience(
    experience_batches: List[Tuple[Dict[str, Union[np.array, list]], int]]
) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Parses experience from the buffers (from dictionaries) into matrices that can be feed into
    neural network in a single pass.

    Args:
        experience_batches: List of dictionaries with trajectories.

    Returns:
        Tuple with matrices:
            * batch [batch_size, observations_dim] of observations,
            * batch [batch_size, observations_dim] of 'next' observations,
            * batch [batch_size, actions_dim] of actions,
            * batch [batch_size, observations_dim] of policies (probabilities or probability densities),
            * batch [batch_size, actions_dim] of rewards,
            * batch [batch_size, actions_dim] of boolean values indicating end of an episode.
    """
    observations = np.concatenate([batch[0]['observations'] for batch in experience_batches], axis=0)
    next_observations = np.concatenate([batch[0]['next_observations'] for batch in experience_batches], axis=0)
    actions = np.concatenate([batch[0]['actions'] for batch in experience_batches], axis=0)
    policies = np.concatenate([batch[0]['policies'] for batch in experience_batches], axis=0)
    rewards = np.concatenate([batch[0]['rewards'] for batch in experience_batches], axis=0)
    dones = np.concatenate([batch[0]['dones'] for batch in experience_batches], axis=0)

    return observations, next_observations, actions, policies, rewards, dones


def is_atari(env_id: str) -> bool:
    """Checks if environments is of Atari type.

    Args:
        env_id: Name (id) of the environment.

    Returns:
        True if its is Atari environment.
    """
    env_spec = [env for env in gym.envs.registry.all() if env.id == env_id][0]
    env_type = env_spec.entry_point.split(':')[0].split('.')[-1]
    return env_type == 'atari'


def get_env_variables(env: gym.Env) -> Tuple[float, int, int, bool, int]:
    """Returns OpenAI Gym environment characteristics.

    Args:
        env: gym.Env object.

    Returns:
        Tuple with:
            * scale of a single action if action's space is continuous, 1 otherwise
            (scale is defined as t, where action can be from interval [-t, t]);
            * dimension of a single action vector;
            * dimension of a single observation vector;
            * boolean value indicating continuous actions space;
            * maximum number of steps in a single episode.
    """
    if type(env.observation_space) == gym.spaces.discrete.Discrete:
        observations_dim = env.observation_space.n
    else:
        observations_dim = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        continuous = False
        actions_dim = env.action_space.n
        action_scale = 1
    else:
        continuous = True
        actions_dim = env.action_space.shape[0]
        action_scale = np.maximum(env.action_space.high, np.abs(env.action_space.low))
    max_steps_in_episode = env.spec.max_episode_steps
    return action_scale, actions_dim, observations_dim, continuous, max_steps_in_episode

