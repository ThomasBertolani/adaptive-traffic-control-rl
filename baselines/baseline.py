import gymnasium as gym
import numpy as np

from envs.traffic_env import Action, Phase


class Baseline:
    def predict(self, observation): ...


class RandomPolicy(Baseline):
    def __init__(self, env, seed: int = 42):
        self.n_actions = env.action_space.n
        self.rng = np.random.default_rng(seed=seed)

    def predict(self, observation):
        return self.rng.choice(self.n_actions)


class CyclePolicy(Baseline):
    def __init__(self, env, cycle_length):
        self.cycle_length = cycle_length
        self.cycle_pos = 0

    def predict(self, observation):
        _, _, _, _, _, timesteps_in_phase = observation
        if timesteps_in_phase == self.cycle_length:
            return Action.SWITCH.value
        else:
            return Action.KEEP.value


class GreedyPolicy(Baseline):
    def __init__(self, env):
        self.env = env

    def predict(self, observation):
        n, e, s, w, phase, _ = observation

        ns = n + s
        ew = e + w

        if ns > ew and phase == Phase.EW.value:
            return Action.SWITCH.value
        elif ew > ns and phase == Phase.NS.value:
            return Action.SWITCH.value
        else:
            return Action.KEEP.value
