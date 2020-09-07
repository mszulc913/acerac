from __future__ import annotations
import pickle
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Type

import numpy as np


@dataclass
class BufferFieldSpec:
    """Specification of the data stored in the replay buffer's data."""
    shape: tuple
    dtype: Union[Type, np.dtype] = np.float32


class ReplayBuffer:
    def __init__(self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec):
        """Stores trajectories.

        Each of the samples stored in the buffer has the same probability of being drawn.

        Args:
            max_size: Maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification.
            obs_spec: BufferFieldSpec with observations space specification.
        """
        self._max_size = max_size
        self._current_size = 0
        self._pointer = 0

        self._actions = self._init_field(action_spec)
        self._obs = self._init_field(obs_spec)
        self._obs_next = self._init_field(obs_spec)
        self._rewards = self._init_field(BufferFieldSpec((), np.float32))
        self._policies = self._init_field(BufferFieldSpec((), np.float32))
        self._dones = self._init_field(BufferFieldSpec((), bool))
        self._ends = self._init_field(BufferFieldSpec((), bool))

    def _init_field(self, field_spec: BufferFieldSpec):
        shape = (self._max_size, *field_spec.shape)
        return np.zeros(shape=shape, dtype=field_spec.dtype)

    def put(
        self, action: Union[int, float, list], observation: np.array, next_observation: np.array,
        reward: float, policy: float, is_done: bool, end: bool
    ):
        """Stores one experience tuple in the buffer.

        Args:
            action: Performed action.
            observation: State in which action was performed.
            next_observation: Next state.
            reward: Earned reward.
            policy: Probability/probability density of executing stored action.
            is_done: True if episode ended and was not terminated by reaching
                the maximum number of steps per episode.
            end: True if episode ended.
        """
        self._actions[self._pointer] = action
        self._obs[self._pointer] = observation
        self._obs_next[self._pointer] = next_observation
        self._rewards[self._pointer] = reward
        self._policies[self._pointer] = policy
        self._dones[self._pointer] = is_done
        self._ends[self._pointer] = end

        self._move_pointer()
        self._update_size()

    def _move_pointer(self):
        if self._pointer + 1 == self._max_size:
            self._pointer = 0
        else:
            self._pointer += 1

    def _update_size(self):
        self._current_size = np.min([self._current_size + 1, self._max_size])

    def get(self, trajectory_len: Optional[int] = None) -> Tuple[Dict[str, np.array], int]:
        """Samples random trajectory. Trajectory length is truncated to the 'trajectory_len' value.
        If trajectory length is not provided, whole episode is fetched. If trajectory is shorter than
        'trajectory_len', trajectory is truncated to one episode only.

        Returns:
            Tuple with dictionary with truncated experience trajectory and first index (always zero in the case of
            this type of a buffer)
        """
        if self._current_size == 0:
            # empty buffer
            return ({
                "actions": np.array([]),
                "observations": np.array([]),
                "rewards": np.array([]),
                "next_observations": np.array([]),
                "policies": np.array([]),
                "dones": np.array([]),
                "ends": np.array([])
            }, -1)
        sample_index = self._sample_random_index()
        start_index, end_index = self._get_indices(sample_index, trajectory_len)

        batch = self._fetch_batch(end_index, sample_index, start_index)
        return batch

    def _fetch_batch(self, end_index: int, sample_index: int, start_index: int) -> Tuple[Dict[str, np.array], int]:

        middle_index = sample_index - start_index \
            if sample_index >= start_index else self._max_size - start_index + sample_index
        if start_index >= end_index:
            buffer_slice = np.r_[0: end_index, start_index: self._max_size]
        else:
            buffer_slice = np.r_[start_index: end_index]

        actions = self._actions[buffer_slice]
        observations = self._obs[buffer_slice]
        rewards = self._rewards[buffer_slice]
        next_observations = self._obs_next[buffer_slice]
        policies = self._policies[buffer_slice]
        done = self._dones[buffer_slice]
        end = self._ends[buffer_slice]

        return ({
            "actions": actions,
            "observations": observations,
            "rewards": rewards,
            "next_observations": next_observations,
            "policies": policies,
            "dones": done,
            "ends": end
        }, middle_index)

    def _get_indices(self, sample_index: int, trajectory_len: int) -> Tuple[int, int]:
        start_index = sample_index
        end_index = sample_index + 1
        current_length = 1
        while True:
            if end_index == self._max_size:
                if self._pointer == 0:
                    break
                else:
                    end_index = 0
                    continue
            if end_index == self._pointer \
                    or self._ends[end_index - 1] \
                    or (trajectory_len and current_length == trajectory_len):
                break
            end_index += 1
            current_length += 1
        return start_index, end_index

    def _sample_random_index(self) -> int:
        """Uniformly samples one index from the buffer."""
        return np.random.randint(low=0, high=self._current_size)

    @property
    def size(self) -> int:
        """Returns number of steps stored in the buffer.

        Returns:
            Size of buffer.
        """
        return self._current_size

    def __repr__(self):
        return f"{self.__class__.__name__}(max_size={self._max_size})"


class PrevReplayBuffer(ReplayBuffer):

    def __init__(self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec):
        """Extends ReplayBuffer to fetch one experience tuple before selected index.
        If selected index indicates first time step in a episode, no additional tuples are attached.
        Policies buffer is replaced to store data in same specification as actions
        (used in algorithms with autocorrelated actions).

        Args:
            max_size: Maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification.
            obs_spec: BufferFieldSpec with observations space specification.
        """
        super().__init__(max_size, action_spec, obs_spec)
        self._policies = self._init_field(action_spec)

    def get(self, trajectory_len: Optional[int] = None) -> Tuple[Dict[str, np.array], int]:
        if self._current_size == 0:
            # empty buffer
            return ({
                "actions": np.array([]),
                "observations": np.array([]),
                "rewards": np.array([]),
                "next_observations": np.array([]),
                "policies": np.array([]),
                "dones": np.array([]),
                "ends": np.array([])
            }, -1)

        start = sample_index = self._sample_random_index()
        trajectory_len_adjusted = trajectory_len
        if self._pointer != sample_index:
            if sample_index > 0 and not self._ends[sample_index - 1]:
                start = sample_index - 1
                trajectory_len_adjusted = trajectory_len + 1
            elif self._current_size == self._max_size and sample_index == 0 and not self._ends[- 1]:
                start = self._max_size - 1
                trajectory_len_adjusted = trajectory_len + 1

        start_index, end_index = self._get_indices(start, trajectory_len_adjusted)
        batch = self._fetch_batch(end_index, sample_index, start_index)
        return batch


class MultiReplayBuffer:

    def __init__(self, max_size: int, num_buffers: int, action_spec: BufferFieldSpec,
                 obs_spec: BufferFieldSpec, buffer_class: type = ReplayBuffer):
        """Encapsulates ReplayBuffers from multiple environments.

        Args:
            max_size: Maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification.
            obs_spec: BufferFieldSpec with observations space specification.
            buffer_class: Class of a buffer to be created.
        """
        self._n_buffers = num_buffers
        self._max_size = max_size

        assert issubclass(buffer_class, ReplayBuffer), "Buffer class should derive from ReplayBuffer"

        self._buffers = [buffer_class(int(max_size / num_buffers), action_spec, obs_spec) for _ in range(num_buffers)]

    def put(self, steps: List[Tuple[Union[int, float, list], np.array, float, np.array, bool, bool]]):
        """Stores experience in the buffers. Accepts list of steps.

        Args:
            steps: List of steps, each of step Tuple consists of: action, observation, reward,
                next_observation, policy, is_done flag, end flag. See ReplayBuffer.put() for format description.
        """
        assert len(steps) == self._n_buffers, f"'steps' list should provide one step (experience) for every buffer" \
                                              f"(found: {len(steps)} steps, expected: {self._n_buffers}"

        for buffer, step in zip(self._buffers, steps):
            buffer.put(*step)

    def get(self, trajectories_lens: Optional[List[int]] = None) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        """Samples trajectory from every buffer.

        Args:
            trajectories_lens: List of maximum lengths of trajectory to be fetched from corresponding buffer.

        Returns:
            List of sampled trajectories from every buffer, see specific buffer class's .get()
            for detailed format description.
        """
        assert trajectories_lens is None or len(trajectories_lens) == self._n_buffers,\
            f"'steps' list should provide one step (experience) for every buffer" \
            f" (found: {len(trajectories_lens)} steps, expected: {self._n_buffers}"

        if trajectories_lens:
            samples = [buffer.get(trajectory_len) for buffer, trajectory_len in zip(self._buffers, trajectories_lens)]
        else:
            samples = [buffer.get() for buffer in self._buffers]

        return samples

    @property
    def size(self) -> List[int]:
        """
        Returns:
            List of buffers' sizes.
        """
        return [buffer.size for buffer in self._buffers]

    @property
    def n_buffers(self) -> int:
        """
        Returns:
            Number of instantiated buffers.
        """
        return self._n_buffers

    def __repr__(self):
        return f"{self.__class__.__name__}(max_size={self._max_size})"

    def save(self, path: str):
        """Dumps the buffer to the disk.

        Args:
            path: File path to the new file.
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> MultiReplayBuffer:
        """Loads the buffer from the disk.

        Args:
            path: Path where the buffer was stored.
        """
        with open(path, 'rb') as f:
            buffer = pickle.load(f)
        return buffer
