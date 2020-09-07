"""
Actor-Critic with Experience Replay and Autocorrelated Actions algorithm.

References:
TBA.
"""

from collections import deque
from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import tf_utils
from algos.base import BaseActor, Critic, BaseACERAgent, GaussianActor
from replay_buffer import BufferFieldSpec, PrevReplayBuffer, MultiReplayBuffer


def get_lambda_1(n: int, alpha: float) -> np.array:
    """Computes Lambda^n_1 matrix.

    Args:
        n: Size of a (square) matrix.
        alpha: Autocorrelation degree.

    Returns:
        Lambda^n_1 matrix.
    """
    lam = np.zeros(shape=(n + 1, n + 1), dtype=np.float32)
    for i in range(n + 1):
        for j in range(i, n + 1):
            lam[i][j] = lam[j][i] = alpha ** abs(i - j) - alpha ** (i + j + 2)
    return lam


def get_lambda_0(n: int, alpha: float) -> np.array:
    """Computes Lambda^n_0 matrix.

    Args:
        n: Size of a (square) matrix.
        alpha: Autocorrelation degree.

    Returns:
        Lambda^n_1 matrix
    """
    lam = np.zeros(shape=(n + 1, n + 1), dtype=np.float32)
    for i in range(n + 1):
        for j in range(i, n + 1):
            lam[i][j] = lam[j][i] = alpha ** abs(i - j)
    return lam


class NoiseGaussianActor(GaussianActor):
    """Actor that produces actions with autocorrelated noise."""

    def __init__(
        self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
        beta_penalty: float, actions_bound: float, tau: int = 2, alpha: float = 0.8,
        num_parallel_envs: int = 1, *args, **kwargs
    ):
        super().__init__(observations_space, actions_space, layers, beta_penalty, actions_bound, *args, **kwargs)

        self._num_parallel_envs = num_parallel_envs
        self._tau = tau
        self._alpha = alpha
        self._noise_dist = tfp.distributions.MultivariateNormalDiag(
            scale_diag=tf.exp(self.log_std),
        )
        self._last_noise = self._sample_noise()
        self._noise_init_mask = tf.ones(shape=(self._num_parallel_envs, 1))

    def _init_noise_buffer(self):
        buffer = deque(maxlen=self._tau)
        for _ in range(self._tau):
            buffer.append(self._noise_dist.sample())
        return buffer

    def _sample_noise(self):
        return self._noise_dist.sample(sample_shape=(self._num_parallel_envs, ))

    def update_ends(self, ends: np.array):
        """Updates noise buffers at the end of an episode."""
        self._noise_init_mask = tf.cast(ends, dtype=tf.float32)

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            scale_diag=tf.exp(self.log_std)
        )

        return dist.prob(actions - mean), dist.log_prob(actions - mean)

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)

        noise_init = self._sample_noise()
        noise_cont = self._alpha * self._last_noise + tf.sqrt(1 - tf.square(self._alpha)) * noise_init
        noise = noise_init * self._noise_init_mask + noise_cont * (1 - self._noise_init_mask)
        self._last_noise = noise
        self._noise_init_mask = tf.zeros_like(self._noise_init_mask)

        actions = mean + noise
        return actions, mean


class ACERAC(BaseACERAgent):
    def __init__(
        self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
        critic_layers: Optional[Tuple[int]], b: float = 3, tau: int = 2, alpha: int = None, *args, **kwargs
    ):
        """Actor-Critic with Experience Replay and Autocorrelated Cctions.

        Args:
            observations_space: Observations' vectors Space.
            actions_space: Actions' vectors Space.
            actor_layers: Number of units in Actor's hidden layers.
            critic_layers: Number of units in Critic's hidden layers.
            b: Density ratio truncating coefficient.
            tau: Update window size.
            alpha: Autocorrelation degree.
        """

        self._tau = tau
        self._alpha = alpha

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._b = b

        self._cov_matrix = tf.linalg.diag(tf.square(tf.exp(self._actor.log_std)))

        self._init_inverse_covariance_matrices()

        self._data_loader = tf.data.Dataset.from_generator(
            self._experience_replay_generator,
            (tf.dtypes.float32, tf.dtypes.float32, self._actor.action_dtype, self._actor.action_dtype, tf.dtypes.float32,
             tf.dtypes.bool, tf.dtypes.int32, tf.dtypes.bool,
             tf.dtypes.float32, self._actor.action_dtype, self._actor.action_dtype)
        ).prefetch(1)

    def _init_inverse_covariance_matrices(self):
        lam0_c_prod_invs = []
        lam1_c_prod_invs = []
        for i in range(0, self._tau):
            lam0 = get_lambda_0(i, self._alpha)
            lam1 = get_lambda_1(i, self._alpha)
            lam0_c_prod = tf_utils.kronecker_prod(lam0, self._cov_matrix)
            lam1_c_prod = tf_utils.kronecker_prod(lam1, self._cov_matrix)
            inv0 = tf.linalg.inv(lam0_c_prod)
            inv1 = tf.linalg.inv(lam1_c_prod)
            lam0_c_prod_invs.append(inv0.numpy())
            lam1_c_prod_invs.append(inv1.numpy())
        self._lam1_c_prod_invs = tf.ragged.constant(lam1_c_prod_invs).to_tensor()
        self._lam0_c_prod_invs = tf.ragged.constant(lam0_c_prod_invs).to_tensor()

    def _init_replay_buffer(self, memory_size: int):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1,)
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            num_buffers=self._num_parallel_envs,
            buffer_class=PrevReplayBuffer
        )

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return NoiseGaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._tau, self._alpha, self._num_parallel_envs,
                self._std, self._tf_time_step
            )

    def _init_critic(self) -> Critic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step)

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, np.array, np.array, bool, bool]
    ]):
        super().save_experience(steps)

        self._actor.update_ends(np.array([[step[5]] for step in steps]))

    def learn(self):
        if self._time_step > self._learning_starts:
            for batch in self._data_loader.take(self._c):
                self._learn_from_experience_batch(*batch)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(
        self, obs: tf.Tensor, obs_next: tf.Tensor, actions: tf.Tensor,
        old_means: tf.Tensor, rewards: tf.Tensor, dones: tf.Tensor,
        lengths: tf.Tensor, is_prev_noise: tf.Tensor,
        prev_obs: tf.Tensor, prev_actions: tf.Tensor, prev_means: tf.Tensor
    ):
        """Performs single learning step.
        Padded tensors are used here and final results are masked out with zeros"""

        is_prev_noise_mask = tf.cast(tf.expand_dims(is_prev_noise, 1), tf.float32)

        c_invs = self._get_c_invs(actions, is_prev_noise_mask)
        eta_repeated, mu_repeated = self._get_prev_noise(actions, is_prev_noise_mask, prev_actions, prev_means, prev_obs)

        with tf.GradientTape(persistent=True) as tape:
            means = self._actor.act_deterministic(obs)
            values, values_next = tf.split(tf.squeeze(self._critic.value(tf.concat([obs, obs_next], axis=0))), 2)
            values_next = values_next * (1 - tf.cast(dones, tf.float32))
            values_first = tf.slice(values, [0, 0], [actions.shape[0], 1])

            actions_flatten = tf.reshape(actions, (actions.shape[0], -1))
            means_flatten = tf.reshape(means, (actions.shape[0], -1))
            old_means_flatten = tf.reshape(old_means, (actions.shape[0], -1))

            actions_repeated = tf.repeat(tf.expand_dims(actions_flatten, axis=1), self._tau, axis=1)
            means_repeated = tf.repeat(tf.expand_dims(means_flatten, axis=1), self._tau, axis=1)
            old_means_repeated = tf.repeat(tf.expand_dims(old_means_flatten, axis=1), self._tau, axis=1)

            # 1, 2, ..., n trajectories mask over repeated action tensors
            actions_mask = tf.expand_dims(
                tf.sequence_mask(
                    tf.range(actions.shape[2], actions_repeated.shape[2] + actions.shape[2], actions.shape[2]),
                    actions_repeated.shape[2],
                    dtype=tf.float32
                ),
                axis=0
            )

            # trajectories shorter than tau mask
            zeros_mask = tf.expand_dims(tf.sequence_mask(lengths, maxlen=self._tau, dtype=tf.float32), 2)

            actions_mu_diff_current = tf.expand_dims(
                (actions_repeated - means_repeated - eta_repeated) * zeros_mask * actions_mask,
                axis=2
            )
            actions_mu_diff_old = tf.expand_dims(
                (actions_repeated - old_means_repeated - mu_repeated) * zeros_mask * actions_mask,
                axis=2
            )
            density_ratio = self._compute_soft_truncated_density_ratio(
                actions_mu_diff_current, actions_mu_diff_old, c_invs
            )

            with tf.name_scope('acerac'):
                tf.summary.scalar('mean_density_ratio', tf.reduce_mean(density_ratio), step=self._tf_time_step)
                tf.summary.scalar('max_density_ratio', tf.reduce_max(density_ratio), step=self._tf_time_step)

            gamma_coeffs = tf.math.cumprod(tf.ones_like(rewards) * self._gamma, exclusive=True, axis=1)
            td_rewards = tf.math.cumsum(rewards * gamma_coeffs, axis=1)

            values_first_repeated = tf.repeat(values_first, self._tau, 1)
            pows = tf.tile(tf.expand_dims(tf.range(1, self._tau + 1), axis=0), [actions.shape[0], 1])
            td = (-values_first_repeated
                  + td_rewards
                  + tf.pow(self._gamma, tf.cast(pows, tf.float32)) * values_next)

            d = td * density_ratio * tf.squeeze(zeros_mask)  # remove artifacts from cumsum and cumprod

            c_mu = tf.matmul(c_invs, tf.transpose(actions_mu_diff_current, [0, 1, 3, 2]))
            c_mu_d = c_mu * tf.expand_dims(tf.expand_dims(d, axis=2), 3)

            c_mu_mean = (tf.reduce_sum(tf.squeeze(c_mu_d), axis=1) / tf.expand_dims(tf.cast(lengths, tf.float32), 1))

            bounds_penalty = tf.scalar_mul(
                    self._actor.beta_penalty,
                    tf.square(tf.maximum(0.0, tf.abs(means) - self._actions_bound))
            )
            bounds_penalty = tf.squeeze(zeros_mask) * tf.reduce_sum(
                bounds_penalty,
                axis=2
            )

            bounds_penalty = tf.reduce_sum(bounds_penalty, axis=1) / tf.cast(lengths, tf.float32)
            actor_loss = tf.matmul(tf.expand_dims(means_flatten, axis=1), tf.expand_dims(tf.stop_gradient(c_mu_mean), axis=2))
            actor_loss = -tf.reduce_mean(tf.squeeze(actor_loss)) + tf.reduce_mean(bounds_penalty)

            d_mean = tf.reduce_sum(d, axis=1) / tf.cast(lengths, tf.float32)
            critic_loss = -tf.reduce_mean(tf.squeeze(values_first) * tf.stop_gradient(d_mean))

        grads_actor = tape.gradient(actor_loss, self._actor.trainable_variables)
        grads_var_actor = zip(grads_actor, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(grads_var_actor)

        with tf.name_scope('actor'):
            tf.summary.scalar(f'batch_actor_loss', actor_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_bounds_penalty', tf.reduce_mean(bounds_penalty), step=self._tf_time_step)

        grads_critic = tape.gradient(critic_loss, self._critic.trainable_variables)
        grads_var_critic = zip(grads_critic, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(grads_var_critic)

        with tf.name_scope('critic'):
            tf.summary.scalar(f'batch_critic_loss', critic_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_value_mean', tf.reduce_mean(values), step=self._tf_time_step)

    def _compute_soft_truncated_density_ratio(
            self, actions_mu_diff_current: tf.Tensor, actions_mu_diff_old: tf.Tensor, c_invs: tf.Tensor
    ):
        exp_current = tf.matmul(
            tf.matmul(actions_mu_diff_current, c_invs),
            tf.transpose(actions_mu_diff_current, [0, 1, 3, 2])
        )
        exp_old = tf.matmul(
            tf.matmul(actions_mu_diff_old, c_invs),
            tf.transpose(actions_mu_diff_old, [0, 1, 3, 2])
        )
        density_ratio = tf.squeeze(tf.exp(-0.5 * exp_current + 0.5 * exp_old))
        return tf.tanh(density_ratio / self._b) * self._b

    def _get_prev_noise(self, actions: tf.Tensor, is_prev_noise_mask: tf.Tensor, prev_actions: tf.Tensor,
                        prev_means: tf.Tensor, prev_obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        current_prev_means = self._actor.act_deterministic(prev_obs)
        alpha_coeffs = tf.pow(
            self._alpha,
            tf.tile(tf.range(1, self._tau + 1, dtype=tf.float32), (prev_actions.shape[0],))
        )
        alpha_coeffs = tf.expand_dims(alpha_coeffs, axis=1)
        mu_diff = (prev_actions - prev_means) * is_prev_noise_mask
        eta_diff = (prev_actions - current_prev_means) * is_prev_noise_mask
        mu = tf.reshape(tf.repeat(mu_diff, self._tau, axis=0) * alpha_coeffs, (actions.shape[0], -1))
        eta = tf.reshape(tf.repeat(eta_diff, self._tau, axis=0) * alpha_coeffs, (actions.shape[0], -1))
        mu_repeated = tf.repeat(
            tf.expand_dims(mu, axis=1),
            self._tau,
            axis=1
        )
        eta_repeated = tf.repeat(
            tf.expand_dims(eta, axis=1),
            self._tau,
            axis=1
        )
        return eta_repeated, mu_repeated

    def _get_c_invs(self, actions: tf.Tensor, is_prev_noise_mask: tf.Tensor) -> tf.Tensor:
        is_prev_noise_mask_repeated = tf.expand_dims(tf.repeat(is_prev_noise_mask, self._tau, axis=0), 1)
        c_invs_0 = tf.tile(self._lam0_c_prod_invs, (actions.shape[0], 1, 1))
        c_invs_1 = tf.tile(self._lam1_c_prod_invs, (actions.shape[0], 1, 1))
        c_invs = c_invs_1 * is_prev_noise_mask_repeated + c_invs_0 * (1 - is_prev_noise_mask_repeated)
        c_invs = tf.reshape(c_invs, (actions.shape[0], self._tau, c_invs.shape[2], -1))
        return c_invs

    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        trajectory_lens = [[self._tau for _ in range(self._num_parallel_envs)] for _ in range(self._batches_per_env)]
        batch = []
        [batch.extend(self._memory.get(trajectories)) for trajectories in trajectory_lens]
        return batch

    def _experience_replay_generator(self):
        """Generates flat batches (matrices) with trajectories.
        All tensors are padded with zeros to match self._tau number of experience tuples in a single trajectory.
        Trajectories are returned in a shape of [batch, self._tau, <obs/actions/etc shape>]
        """
        while True:
            offline_batches = self._fetch_offline_batch()
            obs = np.zeros(shape=(len(offline_batches), self._tau, self._observations_space.shape[0]))
            obs_next = np.zeros(shape=(len(offline_batches), self._tau, self._observations_space.shape[0]))
            actions = np.zeros(shape=(len(offline_batches), self._tau, self._actions_space.shape[0]))
            rewards = np.zeros(shape=(len(offline_batches), self._tau))
            means = np.zeros(shape=(len(offline_batches), self._tau, self._actions_space.shape[0]))
            dones = np.zeros(shape=(len(offline_batches), self._tau))
            lengths = np.zeros(shape=(len(offline_batches)))
            is_prev_noise = np.zeros(shape=(len(offline_batches)))

            prev_obs = np.zeros(shape=(len(offline_batches), self._observations_space.shape[0]))
            prev_actions = np.zeros(shape=(len(offline_batches), self._actions_space.shape[0]))
            prev_means = np.zeros(shape=(len(offline_batches), self._actions_space.shape[0]))

            for i, batch_and_first_index in enumerate(offline_batches):
                batch, first_index = batch_and_first_index
                obs[i * self._tau:, :len(batch['observations'][first_index:]), :]\
                    = batch['observations'][first_index:]
                obs_next[i * self._tau:, :len(batch['next_observations'][first_index:]), :]\
                    = batch['next_observations'][first_index:]
                actions[i * self._tau:, :len(batch['actions'][first_index:]), :]\
                    = batch['actions'][first_index:]
                means[i * self._tau:, :len(batch['policies'][first_index:]), :]\
                    = batch['policies'][first_index:]
                rewards[i * self._tau:, :len(batch['rewards'][first_index:])]\
                    = batch['rewards'][first_index:]
                dones[i * self._tau:, :len(batch['dones'][first_index:])]\
                    = batch['dones'][first_index:]
                is_prev_noise[i] = first_index
                lengths[i] = len(batch['observations'][first_index:])
                prev_obs[i, :] = batch['observations'][0]
                prev_actions[i, :] = batch['actions'][0]
                prev_means[i, :] = batch['policies'][0]

            yield (
                obs,
                obs_next,
                actions,
                means,
                rewards,
                dones,
                lengths,
                is_prev_noise,
                prev_obs,
                prev_actions,
                prev_means,
            )