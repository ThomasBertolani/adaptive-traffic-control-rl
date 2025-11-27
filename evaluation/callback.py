import numpy as np

from algorithms import Agent
from envs import TrafficIntersectionEnv


class EvaluationCallback:
    def __init__(self, eval_env: TrafficIntersectionEnv, eval_freq: int = 1000, n_eval_episodes: int = 3, seed: int = 0, verbose: bool = False):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.logs = []

        self.seed = seed

    def __call__(self, agent: Agent, step):
        if step % self.eval_freq == 0:
            cars_in_queue = 0
            throughput = 0
            timesteps = 0

            for ep in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset(seed=self.seed + ep)
                done = False

                while not done:
                    action = agent.predict(obs)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)

                    cars_in_queue += info["total_queue"]
                    throughput += np.sum(info["departures"])
                    timesteps += 1

                    if terminated or truncated:
                        done = True

            avg_queue_length = cars_in_queue / timesteps
            throughput_per_hour = throughput / self.n_eval_episodes

            self.logs.append(
                {
                    "step": step,
                    "cars_in_queue": avg_queue_length,
                }
            )

            if self.verbose:
                print(
                    f"Step {step}: Avg Q={avg_queue_length:.1f} cars, Flow={throughput_per_hour:.1f} cars/hour"
                )
