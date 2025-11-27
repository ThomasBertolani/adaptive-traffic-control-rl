from gymnasium import ObservationWrapper, spaces
import numpy as np


class AggregateQueues(ObservationWrapper):
    def __init__(self, env, max_size=30, max_timesteps=6):
        super().__init__(env)
        self.max_size = max_size
        self.max_timesteps = max_timesteps
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.max_size, self.max_size, 1, self.max_timesteps]),
            dtype=int,
        )

    def observation(self, observation: np.ndarray):
        n, e, s, w, phase, timesteps = observation.astype(int)
        ns = min(n + s, self.max_size)
        ew = min(e + w, self.max_size)
        return np.array([ns, ew, phase, min(timesteps, self.max_timesteps)])


class BinQueues(ObservationWrapper):
    def __init__(self, env, n_bins=4, max_timesteps=6):
        super().__init__(env)
        self.n_bins = n_bins
        self.max_timesteps = max_timesteps
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.n_bins - 1] * 4 + [1, self.max_timesteps]),
            dtype=int,
        )

        max_queue = 20
        bin_size = max_queue // (self.n_bins - 1)
        self.bins = [x * bin_size for x in range(self.n_bins)]

    def observation(self, observation: np.ndarray):
        n, e, s, w, phase, timesteps = observation.astype(int)

        # subtracting 1 in order to get bins from 0 to self.n_bins, knowing that there cannot be less than 0 zero cars
        bin_queues = np.digitize([n, e, s, w], self.bins) - 1

        return np.concatenate([bin_queues, [phase, min(timesteps, self.max_timesteps)]])


class AggregateBinQueues(ObservationWrapper):
    def __init__(self, env, n_bins=3, max_timesteps=6):
        super().__init__(env)
        self.n_bins = n_bins
        self.max_timesteps = max_timesteps
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.n_bins - 1, self.n_bins - 1, 1, self.max_timesteps]),
            dtype=int,
        )

        max_total_queue = 20
        bin_size = max_total_queue // (self.n_bins - 1)
        self.bins = [x * bin_size for x in range(self.n_bins)]

    def observation(self, observation: np.ndarray):
        n, e, s, w, phase, timesteps = observation.astype(int)

        ns, ew = np.digitize([n + s, e + w], self.bins) - 1

        return np.array([ns, ew, phase, min(timesteps, self.max_timesteps)])
