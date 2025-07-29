# Multi-Agent Simple Spread Environment with DDQN

This repository implements and evaluates a multi-agent reinforcement learning setup using **Double DQN (DDQN)** on the `simple_spread` environment from the [Multi-Agent Particle Environment (MPE)](https://github.com/Farama-Foundation/MPE2/tree/main). The goal is to learn cooperative policies where agents cover landmarks efficiently without colliding.

Fyi, brief notes/comments on the sanity check results can be found [here](https://docs.google.com/document/d/1k-At8SW194tNfY5CPorAJ1PyGzPUWlgp3Uky0ySUyQQ/edit?usp=sharing) 

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/eshikapathak/simple_spread_env.git
cd simple_spread_env 
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv_simple_share
source venv_simple_share/bin/activate   # On Windows use: venv_simple_share\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### Visualize the Environment
To just launch the simple spread environment and render agent and landmark behavior:
```bash
python run_spread.py
```
---
### Train the joint state DDQN agents Q([s1, s2, s3], ai) - no potential function stuff
This script trains three agents independently using DDQN:
```bash
python DDQN_Q_s123_ai/train_jointstate_multiagent_ddqn.py
```
After training, the learned Q-networks are saved in joint_state_models/, and learning metrics are saved in joint_state_logs/.

### Test the trained Agents
Load the Q-networks and evaluate agent behavior over multiple episodes:
```bash
python DDQN_Q_s123_ai/test_jointstate_multiagent_ddqn.py
```
---
### Train the DDQN agents Qi(si, ai) - THIS IS JUST SIMPLE DDQN, NOT THE POTENTIAL FUNCTION PROPOSED STUFF
This script trains three agents independently using DDQN:
```bash
python DDQN_Q_si_ai/train_multiagent_ddqn.py
```
After training, the learned Q-networks are saved in models/, and learning metrics are saved in logs/.

### Test the Trained Agents
Load the Q-networks and evaluate agent behavior over multiple episodes:
```bash
python DDQN_Q_si_ai/test_multiagent_ddqn.py
```
---

### Sanity Check: Environment Setup
Inspect the sizes of agents and landmarks, initial positions, and distances to ensure correct setup:
```bash
python sanity_checks_env_setup.py
```

### Sanity Check: Trained Agent Behavior
Loads the trained Q-networks and prints: Agent positions and velocities, distances to landmarks and other agents, directions to others (e.g. "up-left", "down-right"), Q-values for each action, the selected action and its interpretation
```bash
python sanity_checks_testing_ddqn.py
```







