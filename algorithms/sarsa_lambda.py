import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from algorithms import Agent
from evaluation import EvaluationCallback


class SarsaLambda(Agent):
    def __init__(
        self, env: gym.Env, lambd: float, alpha: float = 0.1, gamma: float = 0.95, seed: int = 42
    ):
        if (
            not isinstance(env.observation_space, Box)
            or not env.observation_space.is_bounded()
            or not isinstance(env.action_space, Discrete)
        ):
            raise ValueError(
                "Sarsa Lambda expects a bounded Box observation space and a Discrete action space"
            )

        if lambd < 0.0 or lambd > 1.0:
            raise ValueError("Lambda must be a float between 0.0 and 1.0")

        self.env = env

        n_states = env.observation_space.high + 1
        self.n_actions = env.action_space.n

        self.q_table = np.zeros(shape=(*n_states, self.n_actions))

        self.lambd = lambd
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.epsilon = self.epsilon_start

        self.env_seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def learn(self, timesteps: int, callback: EvaluationCallback | None = None):
        state, _ = self.env.reset(seed=self.env_seed)
        action = self.choose_action(state)

        traces = np.zeros_like(self.q_table)

        # epsilon_decay = (self.epsilon_min / self.epsilon_start) ** (1 / timesteps)

        for step in range(timesteps):

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            next_action = self.choose_action(next_state)

            target = reward + self.gamma * self.q_table[*next_state, next_action]
            delta = target - self.q_table[*state, action]

            traces[*state, action] = 1

            self.q_table += self.alpha * delta * traces

            traces *= self.gamma * self.lambd

            # self.epsilon *= epsilon_decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if callback is not None:
                callback(self, step)

            if terminated or truncated:
                self.env_seed += 1
                state, _ = self.env.reset(seed=self.env_seed)
                action = self.choose_action(state)
                traces = np.zeros_like(self.q_table)
            else:
                state = next_state
                action = next_action
        
        if callback is not None:
            callback(self, timesteps)

    def predict(self, observation):
        return self.choose_action(observation, greedy=True)

    def choose_action(self, obs, greedy=False):
        if not greedy and self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.choice(self.n_actions)
        else:
            q_values = self.q_table[*obs]
            candidates = np.where(q_values == np.max(q_values))[0]
            return self.rng.choice(candidates)
