import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from algorithms import Agent
from baselines import Baseline
from .callback import EvaluationCallback
from envs import TrafficIntersectionEnv


def evaluate_policy(
    policy: Agent | Baseline, env: TrafficIntersectionEnv, n_runs=10, start_seed=42
):
    all_run_data = []

    for r in range(n_runs):
        obs, _ = env.reset(seed=start_seed + r)

        done = False
        step = 0
        while not done:
            step += 1
            action = policy.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            all_run_data.append(
                {"run": r, "step": step, "cars_in_queue": info["total_queue"]}
            )

    return pd.DataFrame(all_run_data)


def get_learning_curves(
    agent_factory,
    eval_env: TrafficIntersectionEnv,
    training_steps: int = 15000,
    n_runs: int = 3,
    eval_start_seed: int = 42,
):
    all_run_data = []
    for r in range(n_runs):
        agent: Agent = agent_factory(r)
        callback = EvaluationCallback(eval_env, seed=eval_start_seed)
        agent.learn(timesteps=training_steps, callback=callback)

        run_df = pd.DataFrame(callback.logs)
        run_df["run"] = r
        all_run_data.append(run_df)

    final_df = pd.concat(all_run_data, ignore_index=True)
    return final_df


def plot_average_episode(df_policy: pd.DataFrame, title: str, color: str):
    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df_policy, x="step", y="cars_in_queue", color=color, errorbar="sd"
    )

    plt.title(title)
    plt.xlabel("Simulation Steps (10s)")
    plt.ylabel("Total Cars in Queue")
    plt.grid(True, alpha=0.3)
    plt.show()


def run_stability_experiment(
    agent_factory,
    env_factory,
    training_steps: int = 50000,
    n_runs=3,
    label="Algo",
    start_seed=42,
):
    """
    agent_factory: A function that returns a NEW agent instance.
    env_factory: A function that returns a NEW environment instance.
    """
    all_run_data = []

    print(f"--- Starting Stability Experiment for {label} ({n_runs} runs) ---")

    for run_id in range(n_runs):
        print(f"Run {run_id + 1}/{n_runs}...")

        # 1. Seeding for Reproducibility
        seed = start_seed + run_id

        # 2. Create Fresh Envs
        train_env: TrafficIntersectionEnv = env_factory()
        eval_env: TrafficIntersectionEnv = env_factory()

        # 3. Create Fresh Agent
        # We pass the train_env to the factory so the agent attaches to it
        agent: Agent = agent_factory(train_env)

        # 4. Create Fresh Callback
        callback = EvaluationCallback(eval_env, eval_freq=2000)

        # 5. Train
        # The callback populates its own .logs list
        agent.learn(timesteps=training_steps, callback=callback)

        # 6. Harvest Data
        run_df = pd.DataFrame(callback.logs)
        run_df["run_id"] = run_id
        run_df["algorithm"] = label

        all_run_data.append(run_df)

    # Combine all runs into one DataFrame
    final_df = pd.concat(all_run_data, ignore_index=True)
    return final_df
