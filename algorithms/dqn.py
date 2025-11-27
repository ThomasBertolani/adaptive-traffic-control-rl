import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms import Agent
from evaluation import EvaluationCallback


class ReplayBuffer:
    def __init__(self, capacity, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DQN(Agent):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.epsilon = self.epsilon_start

        self.env_seed = seed
        self.rng = np.random.default_rng(seed=seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = QNetwork(input_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(input_dim, self.n_actions).to(self.device)

        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

    def learn(self, timesteps: int, callback: EvaluationCallback | None = None):
        state, _ = self.env.reset(seed=self.env_seed)

        epsilon_decay = (self.epsilon_min / self.epsilon_start) ** (1 / timesteps)

        for step in range(timesteps):
            # self.epsilon = self.epsilon_start - (step/timesteps) * (self.epsilon_start - self.epsilon_min)

            action = self.choose_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            done = terminated
            self.memory.push(state, action, reward, next_state, done)

            self._train_step()

            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon *= epsilon_decay
            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if callback is not None:
                callback(self, step)

            if truncated or terminated:
                self.env_seed += 1
                state, _ = self.env.reset(seed=self.env_seed)
            else:
                state = next_state
        
        if callback is not None:
            callback(self, timesteps)

    def predict(self, observation):
        return self.choose_action(observation, greedy=True)

    def choose_action(self, obs, greedy=False):
        if not greedy and self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.choice(self.n_actions)
        else:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def _train_step(self):
        if len(self.memory) < self.batch_size * 2:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
