# Adaptive Traffic Signal Control using Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-ee4c2c)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom%20Env-black)
![Status](https://img.shields.io/badge/Status-Complete-green)

A comparative study applying **Tabular Reinforcement Learning** and **Deep Reinforcement Learning** to optimize traffic signal control in a non-stationary, stochastic environment. This project demonstrates that Deep RL can outperform state-of-the-art static heuristics by adapting to platoon dynamics in real-time.

---

## 🚦 The Problem
Traditional traffic signals rely on **Fixed Cycle** timers (stable but inefficient) or **Reactive Heuristics** (adaptive but unstable). Real-world traffic arrives in coordinated platoons, and switching lights too frequently incurs a capacity penalty.

**The Goal:** Train an RL agent to minimize queue lengths while maintaining stability, effectively solving the trade-off between reactivity and throughput.

## 🧩 Custom Environment
I designed a custom Gymnasium environment (`TrafficIntersectionEnv`) to model realistic traffic physics:
* **Non-Stationary Inflow:** Traffic arrives in sine-wave gated platoons to simulate upstream signal coordination.
* **Startup Lost Time:** A capacity penalty is applied during phase switches to simulate driver reaction time.
* **Physical Constraints:** Inflow is capped by a safety headway of 1.2s.

## 🧠 Algorithms Implemented
* **Tabular SARSA($\lambda$):** An on-policy method using Eligibility Traces and State Discretization to bridge TD(0) and Monte Carlo returns.
* **Deep Q-Network (DQN):** An off-policy Deep RL method leveraging Experience Replay and Target Networks to handle continuous state spaces.

---

## 📈 View the Full Report

The complete scientific analysis—including hyperparameter sweeps, learning curves, and the "Sim-to-Real" robustness analysis—is detailed in the project notebook.

### [📄 Click here to view the Report (Jupyter Notebook)](report.ipynb)
*(Also available as an HTML export in the repository)*

**Highlight:** The experiments demonstrate that **DQN outperforms the optimized Fixed Cycle baseline by ~25%**, successfully learning to "surf" traffic platoons without the instability associated with standard greedy heuristics.

---

## 🛠️ Usage & Reproduction

If you are curious about the implementation details or want to test the agents with different seeds/parameters, you can clone the repository and run the experiments locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/adaptive-traffic-rl.git](https://github.com/yourusername/adaptive-traffic-rl.git)
cd adaptive-traffic-rl
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the experiments
The entire pipeline is contained in the notebook. You can run it to reproduce all plots and results:
```bash
jupyter notebook Report.ipynb
```

## 📂 Project Structure
```text
.
├── algorithms/
│   ├── agent.py          # Abstract Base Class for agents
│   ├── dqn.py            # Deep Q-Network implementation (PyTorch)
│   └── sarsa_lambda.py   # Tabular SARSA(λ) implementation
├── baselines/
│   └── baseline.py       # Static policies (Random, Greedy, Fixed Cycle)
├── envs/
│   ├── traffic_env.py    # Custom Gymnasium Environment (Physics Engine)
│   └── wrappers.py       # Observation wrappers
├── evaluation/
│   ├── callback.py       # Custom Callback for training logs
│   └── experiments.py    # Reproducibility runners and data aggregators
├── report.ipynb          # Main scientific report and visualization
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 👤 Author
**[Thomas Bertolani]** *Artificial Intelligenge Engineering student @ UniMoRe* [LinkedIn](https://www.linkedin.com/in/thomasbertolani/)